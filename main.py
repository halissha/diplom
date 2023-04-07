import os
import data
from data import ModelData
from arg_parser import ArgumentParser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = ArgumentParser()
    model = parser.get_model()
    data = ModelData(model=model, val_split=parser.get_val_split())
    model.compile(optimizer=parser.get, loss='categorical_crossentropy', metrics=['accuracy'])
    model.train(batch_size=128, epochs=5,
                train_data=(data.X_train, data.y_train),
                validation_data=(data.X_valid, data.y_valid))
    model.evaluate(eval_data=(data.X_valid, data.y_valid), title="CLEAN:")
    model.evaluate(eval_data=(data.X_adv, data.y_valid), title="ADV:")
    model.attack()
    model.evaluate_adversarial()



