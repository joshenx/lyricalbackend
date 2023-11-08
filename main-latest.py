from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from torch import load  # This will be specific to your ML library
from speechbrain.pretrained import EncoderDecoderASR  # Import as per your specific model
from speechbrain.utils.checkpoints import Checkpointer
import speechbrain as sb

import shutil
import os

model_hyperparams = """
# Training params
N_epochs: 20
lr: 0.005

dataloader_options:
    batch_size: 1

# Special tokens and labels
blank_index: 0
num_labels: 41 # 39 phonemes + 1 blank + 1 unknown

# Model parameters
activation: !new:torch.nn.LeakyReLU

compute_features: !new:speechbrain.lobes.features.MFCC

mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: global


model: !new:speechbrain.nnet.containers.Sequential
    input_shape: [null, null, 660]  # input_size
    conv1: !name:speechbrain.nnet.complex_networks.c_CNN.CConv1d
        out_channels: 16
        kernel_size: 3
    nrm1: !name:speechbrain.nnet.complex_networks.c_normalization.CLayerNorm
    act1: !ref <activation>
    conv2: !name:speechbrain.nnet.complex_networks.c_CNN.CConv1d
        out_channels: 32
        kernel_size: 3
    nrm2: !name:speechbrain.nnet.complex_networks.c_normalization.CLayerNorm
    act2: !ref <activation>
    pooling: !new:speechbrain.nnet.pooling.Pooling1d
        pool_type: "avg"
        kernel_size: 3
    RNN: !name:speechbrain.nnet.complex_networks.c_RNN.CLiGRU
        hidden_size: 64
        bidirectional: True

lin: !new:speechbrain.nnet.linear.Linear
    input_size: 256
    n_neurons: !ref <num_labels>
    bias: False

softmax: !new:speechbrain.nnet.activations.Softmax
    apply_log: True

modules:
    compute_features: !ref <compute_features>
    model: !ref <model>
    lin: !ref <lin>
    mean_var_norm: !ref <mean_var_norm>


opt_class: !name:torch.optim.AdamW
    lr: !ref <lr>

compute_cost: !name:speechbrain.nnet.losses.ctc_loss
    blank_index: !ref <blank_index>

per_stats: !name:speechbrain.utils.metric_stats.ErrorRateStats
"""

class PhonemeTokenizer:
    def __init__(self):
        self.tokens = [
            #''' Use this as vocabulary. Do not modify it. '''
            #     Phoneme Example Translation
            # ------- ------- -----------
            '<blank>',  # blank token for CTC decoding
            'AA',  # odd     AA D
            'AE',  # at	AE T
            'AH',  # hut	HH AH T
            'AO',  # ought	AO T
            'AW',  # cow	K AW
            'AY',  # hide	HH AY D
            'B',  # be	B IY
            'CH',  # cheese	CH IY Z
            'D',  # dee	D IY
            'DH',  # thee	DH IY
            'EH',  # Ed	EH D
            'ER',  # hurt	HH ER T
            'EY',  # ate	EY T
            'F',  # fee	F IY
            'G',  # green	G R IY N
            'HH',  # he	HH IY
            'IH',  # it	IH T
            'IY',  # eat	IY T
            'JH',  # gee	JH IY
            'K',  # key	K IY
            'L',  # lee	L IY
            'M',  # me	M IY
            'N',  # knee	N IY
            'NG',  # ping	P IH NG
            'OW',  # oat	OW T
            'OY',  # toy	T OY
            'P',  # pee	P IY
            'R',  # read	R IY D
            'S',  # sea	S IY
            'SH',  # she	SH IY
            'T',  # tea	T IY
            'TH',  # theta	TH EY T AH
            'UH',  # hood	HH UH D
            'UW',  # two	T UW
            'V',  # vee	V IY
            'W',  # we	W IY
            'Y',  # yield	Y IY L D
            'Z',  # zee	Z IY
            'ZH',  # seizure	S IY ZH ER
            '<UNK>', # unknown words
        ]
        self.token_to_id = {} # a dictionary, map phoneme string to id
        self.id_to_token = {} # a dictionary, map phoneme id to string
        self.vocab = set(self.tokens)
        for idx, i in enumerate(self.tokens):
            self.token_to_id[i] = idx
            self.id_to_token[idx] = i


    def encode_seq(self, seq):
        '''
        seq: a list of strings, each string represent a phoneme
        Return a list of numbers. Each representing the corresponding id of that phoneme
        '''
        return [self.token_to_id[x] for x in seq]

    def decode_seq(self, ids):
        '''
        ids: a list of integers, each representing a phoneme's id
        Return a list of strings. Each string represent the phoneme of that id.
        '''
        return [self.id_to_token[x] for x in ids]

    def decode_seq_batch(self, batch_ids):
        '''
        Apply decode_seq to a batch of phoneme ids
        '''
        res = []
        for i in batch_ids:
            res.append(self.decode_seq(i))
        return res
    
class CTCBrain(sb.Brain):
    def on_fit_start(self):
        super().on_fit_start()  # resume ckpt
        self.tokenizer = PhonemeTokenizer()
        self.checkpointer = Checkpointer(self.hparams.checkpoint_dir, {"PER":self.modules})
        self.best_PER = np.inf
        self.metric_fp = self.hparams.output_dir

        self.scheduler = sb.nnet.schedulers.NewBobScheduler(initial_value=self.hparams.lr)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        predictions, lens = predictions
        phns, phn_lens = batch.phn_encoded
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(
                predictions, lens, blank_id=self.hparams.blank_index
            )
            t = phns.tolist()
            out = self.tokenizer.decode_seq_batch(seq)
            tgt = self.tokenizer.decode_seq_batch(t)
            self.per_metrics.append(batch.id, out, tgt)
            with open(self.hparams.result_fn, 'a') as f:
                self.per_metrics.write_stats(f)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss


        elif stage == sb.Stage.VALID:
            PER = self.per_metrics.summarize("error_rate")
            old_lr,new_lr = self.scheduler(metric_value=stage_loss)
            self.hparams.lr = new_lr
            if PER < self.best_PER:
                self.checkpointer.save_and_keep_only()
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % PER)
            print("Old LR: %.7f" % old_lr)
            print("New LR: %.7f" % new_lr)
            train_dict = {"epoch": epoch,"train_loss": self.train_loss,str(stage) +" loss": stage_loss, str(stage) +" PER": PER, "Old LR": old_lr, "New LR": new_lr}
            #train_dict = {"epoch": epoch,"train_loss": self.train_loss,str(stage) +" loss": stage_loss, str(stage) +" PER": PER}
            with open(self.metric_fp / "train_log.txt",'a') as f:
                f.write(str(train_dict)+"\n")


        elif stage == sb.Stage.TEST:
            PER = self.per_metrics.summarize("error_rate")
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % PER)

            test_dict = {str(stage) +" loss": stage_loss, str(stage) +" PER": PER}
            with open(self.metric_fp / "train_log.txt",'a') as f:
                f.write(str(test_dict)+"\n")

temp_dir = 'temp_dir'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

current_directory = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(current_directory, "models", "CKPT.yaml")
MODEL_PATH = os.path.join(current_directory, "models", "PER.ckpt")
hparams = load_hyperpyyaml(model_hyperparams)
device="cpu"

# Load the model (this is a simplified example; your actual loading code may differ)
model = CTCBrain(
    hparams["modules"],
    hparams["opt_class"],
    hparams,
    run_opts={"device": device},)
# Load the state dict from the saved checkpoint
state_dict = load(MODEL_PATH, map_location=model.device)

# If your model components are within a specific namespace, adjust the keys
# This step is important because the state_dict's keys must exactly match your model's architecture
state_dict_corrected = {key.replace('modules.', ''): value for key, value in state_dict.items()}

# Load the state_dict into the model
missing, unexpected = model.modules.load_state_dict(state_dict_corrected, strict=False)

if missing or unexpected:
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

# If you are doing inference, you might want to set the model to evaluation mode
# model.evaluate()

app = FastAPI()

origins = [
    # Set the origins to your frontend's address 
    # (localhost with the appropriate port, or the production URL)
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5173/"
    "http://13.212.192.8:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/model/details")
async def get_model_details():
    # Extract the details you want from your model
    details = {
        "model": model,  # for example
        # ... any other details you want to include
    }
    return details

async def process_audio(file_path: str):
    # This function should handle the audio processing and prediction
    # Dummy function (replace with your model's prediction logic)
    result = model(file_path)
    return result

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        if file.filename.split('.')[-1] != "wav":
            return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

        temp_dir = 'temp_dir'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        file_path = os.path.join(temp_dir, 'temp_audio.wav')

        with open(file_path, 'wb') as buffer:
            # You can use shutil to save the uploaded file to the path
            shutil.copyfileobj(file.file, buffer)

        print(model.predict(file_path))

        # Here, add your logic to evaluate the .wav file using your model.
        # Ensure you replace this with your actual model evaluation logic.
        result = model(file_path)  # This is a hypothetical method. Replace with your actual method.

        return JSONResponse(content={"prediction": result}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)