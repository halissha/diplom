import os
from data import ModelData
from arg_parser import ArgumentParser
from output.printer import print_attack_results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = ArgumentParser()
    model = parser.get_model()
    model_data = ModelData(dataset=parser.get_dataset_name())
    model_data.split_data(val_split=parser.get_val_split())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.train(batch_size=128, epochs=5,
                train_data=(model_data.X_train, model_data.y_train),
                validation_data=(model_data.X_valid, model_data.y_valid),
                weights=parser.get_weights())
    model.evaluate(eval_data=(model_data.X_test, model_data.y_test), title="CLEAN:")
    model_data.generate_adversarial_data(model=model, attack=parser.get_attack(), data=model_data.X_test)
    model.evaluate(eval_data=(model_data.X_adv, model_data.y_test), title="ADV:")
    print_attack_results(model=model, data=model_data, attack=parser.get_attack(), dataset=parser.get_dataset_name())



