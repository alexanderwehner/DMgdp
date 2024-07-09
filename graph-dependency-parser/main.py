from min_train import train
from simplemodel import MlpParsingModel, DozatManningParsingModel
from util import Config

UD_DATASET = "en_ewt"
config = Config.load("config.yml")

modelchoice = config.modelchoice

if modelchoice == "mlp":
    model = MlpParsingModel(roberta_id="xlm-roberta-base", dropout=config.dropout, activation=config.transformer_activation)
elif modelchoice == "dm":
    model = DozatManningParsingModel(roberta_id="xlm-roberta-base", dropout=config.dropout, activation=config.transformer_activation)


train(config, UD_DATASET, model)
