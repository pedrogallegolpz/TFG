from torch import nn

def get_output_last_conv(module):
    out_features_channels = -1  # output
    for m in module.children():
        # Test the type
        if isinstance(m, nn.Conv2d):
            # If it's Conv2d
            out_features_channels = m.out_channels
        elif isinstance(m, nn.Sequential):
            # If it's sequential
            out_features_channels_new = get_output_last_conv(m)
            if out_features_channels_new != -1:
                out_features_channels = out_features_channels_new

    return out_features_channels


def remove_modules_type(module, types_module):
    """
    Parameters:
    -----------
    module: nn.Module
      Module where we want to remove {type_module} modules.

    types_module: list of class which inherits from nn.Module
      Types to remove.

    Description:
    ------------
    This function remove all {type_module} modules from
    a first module: {module}

    Return:
    -------
    List of nn.Module
    """
    presence_type = False
    mod_without_type = []
    
    # Prueba:
    in_features_channels = []  # input of the module we are removing
    for m in module.children():
        # Test the type
        for type_module in types_module:
            if isinstance(m, type_module) and not presence_type:
                presence_type = True
                try:
                    # If it's Linear
                    in_features_channels.append(m.in_features)
                except:
                    try:
                        # If it's Conv2d
                        in_features_channels.append(m.in_channels)
                    except:
                        # Other cases are not considered
                        in_features_channels = in_features_channels

                break

        if presence_type:
            return mod_without_type, in_features_channels
        
        elif isinstance(m, nn.Sequential):
            # If it's sequential
            modules_removed, in_features_channels_new = remove_modules_type(m, [nn.Linear])
            in_features_channels = in_features_channels + in_features_channels_new

            if len(modules_removed)>0:
                mod_without_type.append(nn.Sequential(*modules_removed))
        else:
            mod_without_type.append(m)
            

    return mod_without_type, in_features_channels
