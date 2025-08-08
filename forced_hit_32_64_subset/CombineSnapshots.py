import numpy as np
import h5py as h5
import os
from random import shuffle
import time

if __name__ == "__main__":
    # Settings ----------
    # Number of boxes per stage/chunk
    num_boxes_stage = 8896
    Delta = 2 #Downsampling factor
    # Variables to load
    real_vars = ["/flow/U", "/flow/V", "/flow/W"] #["/flow/u", "/flow/v", "/flow/w", "/ps/ps_01"]
    # If you place the script in the boxes directory (with subdirs DNS and LES, you can leave this unchanged)
    home_dir = os.getcwd()
    LES_target_dir = os.getcwd() + "/LES/"
    DNS_target_dir = os.getcwd() + "/DNS/"
    LES_dir = os.getcwd() + "/LES/"
    DNS_dir = os.getcwd() + "/DNS/"
    # Code ---------
    # Get all h5 files that start with boxes and are in the DNS directory:
    f_list = []
    for f_name in os.listdir(DNS_dir):
        if f_name.endswith(".h5") and f_name.startswith("Boxes"):
                f_list.append(f_name)
    # Create random indices; create list of integers
    random_idxs = [i for i in range(len(f_list))]
    # Shuffle integers
    shuffle(random_idxs)
    # Go to DNS directory
    os.chdir(DNS_dir)
    # Open first file to extract number of boxes/file and box dimensions
    h5_dns = h5.File(f_list[0], "r")
    num_boxes, box_dim, _, _ = np.shape(h5_dns[real_vars[0]])
    h5_dns.close()
    # Calculate number of files/stage
    files_per_stage = int(np.floor(num_boxes_stage / (num_boxes)))
    print("Files per stage: {}".format(str(files_per_stage)))
    # Make lists with files in each stage
    stage_lists = []
    stage_list_tmp = []
    for idx, r_idx in enumerate(random_idxs):
        if (idx % files_per_stage - 1) == 0:
            stage_lists.append(stage_list_tmp)
            stage_list_tmp = []
        stage_list_tmp.append(r_idx)
    # Don't forget the last one (incomplete)
    stage_lists.append(stage_list_tmp)
    # Compute number of boxes/stage
    boxes_per_stage = files_per_stage * (num_boxes)
    print("Boxes per stage: {}".format(str(boxes_per_stage)))
    # Loop through all stage lists
    for stage_idx, stage in enumerate(stage_lists):
        # Skip index 0
        if stage_idx == 0:
            continue
        t_0 = time.time()
        print("Working on stage {}".format(str(stage_idx)))
        # Assign empty ndarray
        combined_boxes = np.ndarray((boxes_per_stage, box_dim, box_dim, box_dim, len(real_vars)))
        # Create directories
        os.makedirs(DNS_target_dir + "stage_reduced_" + str(stage_idx))
        os.makedirs(LES_target_dir + "stage_reduced_" + str(stage_idx))
        box_no = 0
        print("len stage: {}".format(str(len(stage))))
        # Loop through all files in the stage
        for f_idx in stage:
            os.chdir(DNS_dir)
            dns_f = f_list[f_idx]
            print("Working on file {}".format(dns_f))
            # Read file
            h5_dns = h5.File(dns_f, "r")
            # Extract boxes and assign to ndarray
            for var_idx, var in enumerate(real_vars):
                combined_boxes[box_no:box_no + num_boxes, :, :, :, var_idx] = h5_dns[var][()]
            h5_dns.close()
            # Update number of boxes, so they are not overwritten
            box_no += num_boxes
        os.chdir(DNS_target_dir + "stage_reduced_" + str(stage_idx))
        # Now write ndarray to single .h5 file in new directory
        print("Writing corresponding .h5 combined file...")
        with h5.File("Boxes_Stage" + str(stage_idx) + ".h5", "w") as f:
            for var_idx, var in enumerate(real_vars):
                f.create_dataset(var, data=combined_boxes[:, :, :, :, var_idx])
        os.chdir(LES_dir)
        box_no = 0
        combined_boxes_les = np.ndarray((boxes_per_stage, int(box_dim/Delta), int(box_dim/Delta), int(box_dim/Delta), len(real_vars)))
        for f_idx in stage:
            les_f = "_".join([f_list[f_idx].split("_")[0]] + ["DownsampledFiltered"] + f_list[f_idx].split("_")[1:])
            print("Working on file {}".format(les_f))
            h5_les = h5.File(les_f, "r")
            for var_idx, var in enumerate(real_vars):
                combined_boxes_les[box_no:box_no + num_boxes, :, :, :, var_idx] = h5_les[var][()]
            h5_les.close()
            box_no += num_boxes
        os.chdir(LES_target_dir + "stage_reduced_" + str(stage_idx))
        print("Writing corresponding .h5 combined file...")
        with h5.File("Boxes_DownsampledFiltered_Stage" + str(stage_idx) + ".h5", "w") as f:
            for var_idx, var in enumerate(real_vars):
                f.create_dataset(var, data=combined_boxes_les[:, :, :, :, var_idx])
        delta_t = time.time() - t_0
        print("Stage took {}s".format(str(round(delta_t, 2))))

