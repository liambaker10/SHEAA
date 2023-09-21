# beta, error mitigation
import torch
import torch.nn as nn

class Defender():
    """This class defines the error mitigation schemes (not exhaustive). Add your custom error mitigation methods as classmethod here.
    """    

    @classmethod
    def activation_clipping(cls, model, **kwargs) -> None:
        """This method adds forward hooks to each named parameter to check if there are invalid activation values.
        """  
        def clamp_output(module, input, output):
            output.nan_to_num_(nan = 0.0)
            output.clamp_(min = kwargs['min'], max = kwargs['max'])
        
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_layer_name = name
                break

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name != last_layer_name:
                module.register_forward_hook(clamp_output)
    
    @classmethod
    def sbp(cls, error_maps: dict, **kwargs) -> None:
        """Sanitize the error map according to selected bit positions for error mitigation.

        Args:
            error_map (dict): The error map for selective bit protection
        """        
        error_maps_ = error_maps.copy()
        sbp_mask = ~(torch.tensor([1 << i for i in kwargs['protected_bits']]).sum())
        for param_name, param in error_maps_.items():
            error_maps_[param_name] = param & sbp_mask
        return error_maps_
