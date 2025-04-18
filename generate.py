import torch
import config
import model
import dataSet
from processor.processUtils import decode_midi,encode_midi

config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

music_model = model.MusicTransformer(
    embedding_dim=config.EMBEDDING_DIM,
    vocab_size=config.VOCAB_SIZE,
    num_layer=config.NUM_LAYERS,
    max_seq=config.MAX_SEQ_LEN,
    dropout=config.DROP_OUT,
).to(config.DEVICE)
music_model.load_state_dict(torch.load("music_model.pth"))
music_model.to(config.DEVICE)

dataset = dataSet.MIDIDataset(config.SAVE_DIR)
input, _ = dataset.slide_seq2seq_batch(1, config.SEQ_LEN, predict=config.SLIDE_LEN)
#或者自己导
#input=encode_midi("file.mid")
input = torch.from_numpy(input).contiguous().to(config.DEVICE, non_blocking=True, dtype=torch.long)

melody = music_model.generate(input, config.TARGET_LEN)

decode_midi(melody,"melody.mid")