
from model.diffusion.common import ModelMeanType, ModelVarType, LossType
from model.diffusion.gaussian_diffusion import GaussianDiffusion, SpacedDiffusion


ModelMeanTypeDict= {
  'PREVIOUS_X': ModelMeanType.PREVIOUS_X,
  'START_X': ModelMeanType.START_X,
  'EPSILON': ModelMeanType.EPSILON
}

ModelVarTypeDict= {
  'LEARNED': ModelVarType.LEARNED,
  'FIXED_SMALL': ModelVarType.FIXED_SMALL,
  'FIXED_LARGE': ModelVarType.FIXED_LARGE,
  'LEARNED_RANGE': ModelVarType.LEARNED_RANGE
}

LossTypeDict= {
  'MSE': LossType.MSE,
  'RESCALED_MSE': LossType.RESCALED_MSE,
  'KL': LossType.KL,
  'RESCALED_KL': LossType.RESCALED_KL
}

def initialize_diff_model(betas, config, model):
    model_var_type = ModelVarTypeDict[config.diffusion.model_var_type]
    model_mean_type = ModelMeanTypeDict[config.diffusion.model_mean_type]
    loss_type = LossTypeDict[config.diffusion.loss_type]

    available_models = ["GaussianDiffusion", "SpacedDiffusion"]

    if model == "GaussianDiffusion":
        diffusion = GaussianDiffusion(config,
                                      betas=betas,
                                      model_var_type=model_var_type,
                                      model_mean_type=model_mean_type,
                                      loss_type=loss_type)
    elif model == "SpacedDiffusion":
        diffusion = SpacedDiffusion(config,
                                    betas=betas,
                                    model_var_type=model_var_type,
                                    model_mean_type=model_mean_type,
                                    loss_type=loss_type)

    else:
        raise ValueError(f"Unknown diffusion model: {config.diffusion.model}. Supported: {available_models}")
    return diffusion
