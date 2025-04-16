from matplotlib import pyplot as plt
import numpy as np

def plot_efforts(logistic_efforts, efforts_gpt, feature_list, ncols, nrows, prefix, sz=(12,4), mlp=False):
    """
    plot theoretic efforts and gpt efforts for each feature
    """
    n = len(feature_list)
    # x = np.arange(1000)
    fig, ax = plt.subplots(nrows, ncols, figsize=sz)
    leg=False
    if ncols * nrows > n:
        fig.delaxes(ax[nrows-1][ncols-1])
    for i in range(n):
        if nrows == 1:
            if mlp:
                ax[i].hist(logistic_efforts[:,i], bins=30, color='blue', alpha=0.5, label='Theoretical effort')
            else:
                ax[i].axvline(x=logistic_efforts[i], color='blue', label='Theoretical effort')
            # clip efforts gpt to be between -2 and 2
            efforts_gpt[:,i] = np.clip(efforts_gpt[:,i], -2, 2)
            ax[i].hist(efforts_gpt[:,i], bins=30, color='red', alpha=0.5, label='GPT')
            ax[i].set_title(feature_list[i], fontsize=12)
            ax[i].set_xticks(np.arange(-2,2,0.5))
            ax[i].set_yticks(np.arange(0,1000,200))
            ax[i].set_xlabel('Effort')
            ax[i].set_ylabel('Frequency')
            if not leg:
                ax[i].legend(handlelength = 0.5)
                leg=True
        else:
            if mlp:
                ax[i//ncols, i%ncols].hist(logistic_efforts[:,i], bins=30, color='blue', alpha=0.5, label='Theoretical effort')
            else:
                # logistic efforts plot as a vertical line
                ax[i//ncols, i%ncols].axvline(x=logistic_efforts[i], color='blue', label='Theoretical effort')
            # plot the distribution of GPT efforts
            ax[i//ncols, i%ncols].hist(efforts_gpt[:,i], bins=30, color='red', alpha=0.5, label='GPT')
            ax[i//ncols, i%ncols].set_title(feature_list[i], fontsize=12)
            ax[i//ncols, i%ncols].set_xticks(np.arange(-2,2,0.5))
            ax[i//ncols, i%ncols].set_yticks(np.arange(0,1000,200))
            ax[i//ncols, i%ncols].set_xlabel('Effort')
            ax[i//ncols, i%ncols].set_ylabel('Frequency')
            if not leg:
                ax[i//ncols, i%ncols].legend(handlelength = 0.5)
                leg=True
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}_efforts.pdf")
    plt.show()

def plot_scores(data1, data2, data3, prefix):
    plt.figure(figsize=(6, 3))

    # Define 10 bins between 0 and 1 for the two histograms, ensuring they do not overlap
    bins = np.linspace(0, 1, 20)
    bin_width = bins[1] - bins[0]
    bins_shifted = bins + bin_width / 2  # Shift the bins for the second histogram to avoid overlap

    # Plot the histogram of the data1 (Theoretical) as hollow white
    plt.hist(data1, bins=bins,  alpha=1.0, color='white',  histtype='bar', edgecolor='black', width=bin_width / 2.5, label='Theoretical')
    
    # Plot the histogram of the data2 (GPT) as solid black
    plt.hist(data2, bins=bins_shifted[:-1], alpha=1.0, color='black', label='GPT', histtype='bar', edgecolor='black', width=bin_width / 2.5)
    
    # Compute the histogram for data3 (Original)
    hist3, edges3 = np.histogram(data3, bins=bins)
    
    # Plot the binned data3 as a line plot with stars
    bin_centers = edges3[:-1]
    plt.plot(bin_centers, hist3, 'g*-', linewidth=1, label='Original')
    
    plt.legend(handlelength=0.5)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'plots/{prefix}_score_dist.pdf')
    plt.show()

        

