import time
import torch
import torch.nn as nn

class Injector():
    valid_dtypes = [torch.float, ]
    valid_error_models = ['bit', 'value']
    valid_mitigations = ['None', 'SBP', 'clip']

    @classmethod
    def _error_map_generate(cls, injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, p: float, seed) -> torch.Tensor:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        error_map = (2 * torch.ones((*injectee_shape, dtype_bitwidth), dtype=torch.int, device=device)) ** torch.arange(0, dtype_bitwidth, dtype=torch.int, device=device).flip(dims=(-1, )).expand((*injectee_shape, dtype_bitwidth))
        filter = (p * nn.functional.dropout(torch.ones_like(error_map, dtype=torch.float, device=device), 1 - p)).int()
        error_count = filter.sum(dim = -1)
        error_map = (filter * error_map).sum(dim = -1).int()
        return error_map, error_count


    def __init__(self,
                 p: float = 1e-10,
                 dtype: torch.dtype = torch.float,
                 param_names: list = ['weight'],
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False,
                 error_model = 'bit',
                 mitigation = 'None',
                 seed = 0,
                 ) -> None:

        self.p = p
        self.dtype = dtype
        self.param_names = param_names
        self.device = device
        self.verbose = verbose
        self.error_model = error_model
        self.mitigation = mitigation
        self.seed = seed
        self._argument_validate()
        self._dtype_bitwidth = torch.finfo(self.dtype).bits
        self._error_maps = {}
        self._error_count = {}
        self._check_list = {}
        
    def _argument_validate(self) -> None:
        if self.p <= 0 or self.p >= 1:
            raise ValueError('Invalid probability of error injection.')
        if self.dtype not in Injector.valid_dtypes:
            raise ValueError('Invalid data types.')
        if self.error_model not in Injector.valid_error_models:
            raise ValueError('Unknown error model.')
        if self.mitigation not in Injector.valid_mitigations:
            raise ValueError('Unknown mitigation method.')
        if self.verbose == True:
            print('Injector initialized.\nError probability:', self.p)
            print('Data type:', self.dtype)
            print('Error model:', self.error_model)

    def _error_map_allocate(self, model: nn.Module) -> None:
        """Iterative through model parameters and allocate the error maps for injection.
        Args:
            model (nn.Module): The target model for error injection.
        """
        if type(model) == dict:
            for param_name, param in model.items():
                for each_param in self.param_names:
                    if each_param in param_name:
                        injectee_shape = param.shape
                        self._error_maps[param_name], self._error_count[param_name] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p, self.seed)
                        break
        else:
            for param_name, param in model.named_parameters():
                for each_param in self.param_names:
                    if each_param in param_name:
                        injectee_shape = param.shape
                        self._error_maps[param_name], self._error_count[param_name] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p, self.seed)
                        break
                    
    def profiling(self, model: nn.Module) -> None:

        for param_name, param in model.named_parameters():
            self._check_list[param_name] = (param.max(), param.min())
        if self.verbose == True:
            name_params = self._check_list.keys()
            print('All layers have been profiled: %i layers in total'%len(name_params))

    def inject(self, model: nn.Module) -> None:
        """Injecting the errors into the model
        Args:
            model (nn.Module): The target model.
        """
        start_time = time.time()
        self._error_map_allocate(model)
        error_count_number = 0
        param_count_number = 0

        if type(model) == dict:
            for param_name, param in model.items():
                if param_name in self._error_maps.keys():
                    error_mask = self._error_maps[param_name]
                    error_count_number += self._error_count[param_name].sum()
                    param_count_number += self._error_maps[param_name].numel()
                    param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
        else:
            for param_name, param in model.named_parameters():
                if param_name in self._error_maps.keys():
                    error_mask = self._error_maps[param_name]
                    error_count_number += self._error_count[param_name].sum()
                    param_count_number += self._error_maps[param_name].numel()
                    param.data = (param.view(torch.int) ^ error_mask).view(torch.float)

        if self.verbose == True:
            injected_params = self._error_maps.keys()
            print('The following parameters have been injected: %i layers in total'%len(injected_params))
            print('Total number of errors injected:', error_count_number)
            print('Total number of parameters:', param_count_number)
            print('Time spent on error injection (second):', time.time() - start_time)

    def correct(self, model: nn.Module) -> None:
        """Correcting the errors in the model
        Args:
            model (nn.Module): The target model.
        """
        start_time = time.time()
        self._error_map_allocate(model)
        error_count_number = 0
        param_count_number = 0
        # print(self._error_maps.keys())
        for param_name, param in model.named_parameters():
            if param_name in self._error_maps.keys():
                exception_mask = torch.isnan(param) + torch.isinf(param)
                error_count_number += exception_mask.sum()
                param_count_number += exception_mask.numel()
                #exceptional SDCs
                param.nan_to_num_(nan = 0.0)
                #soft SDCs
                with torch.no_grad():
                    ceil, floor = self._check_list[param_name]
                    # print(self._check_list[param_name], ceil, floor, param.device)
                    torch.clamp_(param, max = ceil, min = floor)

        if self.verbose == True:
            corrected_params = self._error_maps.keys()
            print('The following parameters have been corrected: %i layers in total'%len(corrected_params))
            print('Total number of exceptional errors:', error_count_number)
            print('Total number of parameters:', param_count_number)
            print('Time spent on error injection (second):', time.time() - start_time)
        # return model

class Defender():
    """This class defines the error mitigation schemes (not exhaustive). Add your custom error mitigation methods as classmethod here.
    """    

    @classmethod
    def activation_limitation(cls, model, **kwargs) -> None:
        """This method adds forward hooks to each named parameter to check if there are invalid activation values.
        """  
        def clamp_output(module, input, output):
            output.nan_to_num_(nan = 0.0)
            output.clamp_(min = kwargs['min'], max = kwargs['max'])

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
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