import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def make_log_plots(input_log_file):

    # Load TensorBoard log
    ea = event_accumulator.EventAccumulator(input_log_file)
    ea.Reload()

    var_log_dict = {'total_loss':'loss/total',
                    'proj_loss':'loss/proj',
                    'loss_clust':'loss/clust_only',
                    'loss_entropy':'loss/entropy',
                    'lr':'lr/train'}
    
    var_name_dict = {'total_loss':'loss_total',
                     'proj_loss':'loss_proj',
                     'loss_clust':'loss_clust_only',
                     'loss_entropy':'loss_entropy',
                     'lr':'lr'}
    
    ## Loop over keys
    for key in var_log_dict:

        ## Get the xaxis
        xvals = [e.step for e in ea.Scalars(var_log_dict[key])]
        yvals = [e.value for e in ea.Scalars(var_log_dict[key])]
    
        # Plot
        plt.figure(figsize=(6,4))
        plt.plot(xvals, yvals, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(var_log_dict[key])
        plt.tight_layout()
        
        # Save instead of showing
        output_file = "log_"+var_name_dict[key]+".png"
        plt.savefig(output_file, dpi=300)
        print(f"Saved plot to {output_file}")
        
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("An input file name must be provided as an argument!")
        sys.exit()
                                     
    make_log_plots(sys.argv[1])
        
