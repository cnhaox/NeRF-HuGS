import models.nerfacto as nerfacto
import models.nerf as nerf

model_config_dict = {
    'nerfacto': nerfacto.ModelConfig,
    'nerf': nerf.ModelConfig,
}

model_dict = {
    'nerfacto': nerfacto.Model,
    'nerf': nerf.Model,
}

criterion_dict = {
    'nerfacto': nerfacto.Loss,
    'nerf': nerf.Loss,
}