import utils
model = utils.DModel() 
model_path = 'tfmodel/VGG_ILSVRC_16_layers.npy'
model.load_model(model_path)
print model.data_dict
print model.data_dict.keys()