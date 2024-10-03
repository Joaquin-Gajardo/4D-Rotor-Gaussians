import json
import os

import pandas as pd

def summarize_results(base_folder, model_name='splatfacto'):
    # List to store results
    results = []

    # Iterate through each subfolder in the base folder
    for scene_folder in os.listdir(base_folder):
        scene_path = os.path.join(base_folder, scene_folder)

        # Skip if scene folder ends with a number (e.g. "scene2"), assumes final runs are without numbers
        if scene_path[-1].isdigit() or scene_path.split('-')[-1] == 'debug':
            continue
        
        # Check if it's a directory
        if os.path.isdir(scene_path):
            model_folder = os.path.join(scene_path, model_name)
            last_run = sorted(os.listdir(model_folder))[-1]
            run_path = os.path.join(model_folder, last_run)
            json_path = os.path.join(run_path, 'results.json')
            
            # Check if the JSON file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract values
                data = data.get('metrics', {})
                
                for timestep in data:
                    timestep.update({'Scene': scene_folder})
                
                results.extend(data)
            else:
                print(f"Warning: No JSON file found for scene {scene_folder}")

    # Create a DataFrame
    df = pd.DataFrame(results).sort_values('Scene').reset_index(drop=True)
    return df

base_folder = "outputs/WAT"
df = summarize_results(base_folder)
df.insert(0, 'Scene', df.pop('Scene'))
df = df.rename(columns={'LPIPS-VGG': 'LPIPS (vgg)'})

print(df.to_string())

#summary_table.to_csv("dynerf_realtime4DGS_results_summary.csv")
df.to_csv("WAT_rotor4DGS_results_summary.csv", index=False)

df = df.sort_values(by=['Scene', 'Timestep'], ascending=[True, True]).reset_index(drop=True)

# Get dataframe with mean in time
df_timeaverage = df.groupby('Scene').agg({
                                'PSNR': 'mean',
                                'SSIM': 'mean',
                                'LPIPS (vgg)': 'mean'
                            }).reset_index()

# Calculate overall mean
mean_row = pd.DataFrame({
    'Scene': ['Overall Mean'],
    'PSNR': [df_timeaverage['PSNR'].mean()],
    'SSIM': [df_timeaverage['SSIM'].mean()],
    'LPIPS (vgg)': [df_timeaverage['LPIPS (vgg)'].mean()]
})

# Concatenate the mean row to the original DataFrame
df_timeaverage = pd.concat([df_timeaverage, mean_row], ignore_index=True)

print(df_timeaverage.to_string())

df_timeaverage.to_csv("WAT_rotor4DGS_results_summary_timeaveraged.csv", index=False)
