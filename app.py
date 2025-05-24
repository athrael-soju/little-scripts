import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

hf_repo = "nvidia/C-RADIOv2-B"

try:
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()
    
    # Only move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    image = Image.open('atlas.png').convert('RGB')
    
    # Get model's supported resolutions
    min_res_step = getattr(model, 'min_resolution_step', 32)  # Default to 32 if not available
    
    # Resize image to nearest supported resolution
    width, height = image.size
    new_height = ((height + min_res_step - 1) // min_res_step) * min_res_step
    new_width = ((width + min_res_step - 1) // min_res_step) * min_res_step
    
    # Resize to supported dimensions
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    pixel_values = image_processor(images=image, return_tensors='pt', do_resize=False).pixel_values
    pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
        summary, features = model(pixel_values)
    
    print("Summary shape:", summary.shape if hasattr(summary, 'shape') else type(summary))
    print("Features shape:", features.shape if hasattr(features, 'shape') else type(features))
    print("Summary:", summary)
    print("Features:", features)
    
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative loading method...")
    
    # Alternative approach using direct model loading
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_repo, config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        
        image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        image = Image.open('atlas.png').convert('RGB')
        
        # Try to handle resolution requirements
        width, height = image.size
        
        # Try resizing to common resolutions that models typically support
        for target_size in [448, 512, 336, 384]:
            try:
                image_resized = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                pixel_values = image_processor(images=image_resized, return_tensors='pt', do_resize=False).pixel_values
                pixel_values = pixel_values.to(device)
                
                with torch.no_grad():
                    output = model(pixel_values)
                    if isinstance(output, tuple):
                        summary, features = output
                    else:
                        summary = output
                        features = None
                
                print(f"Alternative method successful with resolution {target_size}x{target_size}!")
                print("Summary shape:", summary.shape if hasattr(summary, 'shape') else type(summary))
                if features is not None:
                    print("Features shape:", features.shape if hasattr(features, 'shape') else type(features))
                print("Summary:", summary)
                if features is not None:
                    print("Features:", features)
                break
                
            except Exception as res_e:
                print(f"Failed with {target_size}x{target_size}: {res_e}")
                continue
        else:
            raise Exception("All resolution attempts failed")
        
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        print("You may need to use a different transformers version or contact the model authors.")
