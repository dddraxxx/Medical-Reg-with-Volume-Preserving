def identify_axes(ax_dict, fontsize=48):
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def plot_grid(ax, flow, factor=10):
    """ Plot the grid generated by a flow. The displacement can be too small, so we add a scale factor"""
    grid = factor * flow[:, ::8, ::8]
    lin_range = np.linspace(0, 512, 64)
    x, y = np.meshgrid(lin_range, lin_range)
    x = x + grid[0, ...]
    y = y + grid[1, ...]
    y = y

    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color='black', linewidths=0.8))
    ax.add_collection(LineCollection(segs2, color='black', linewidths=0.8))
    ax.autoscale()


def generate_plots(fixed, moving, warped, flows, train_loss, val_loss, reg_loss, epoch):
    """ Save some images and plots during training"""
    moving = moving.detach().cpu().numpy()
    fixed = fixed.detach().cpu().numpy()
    warped = [w.detach().cpu().numpy() for w in warped]
    flows = [f.detach().cpu().numpy() for f in flows]

    fig = plt.figure(constrained_layout=True, figsize=(4 * 5, 4 * 3))
    ax_dict = fig.subplot_mosaic("""
                                 FABCD
                                 LGHIE
                                 MKJWX
                                 """)

    ax_dict['F'].imshow(moving[0, 0, ...], cmap='gray')
    ax_dict['F'].set_title('Moving')

    ax_dict['W'].imshow(fixed[0, 0, ...], cmap='gray')
    ax_dict['W'].set_title('Fixed')

    for i, ax_name in enumerate(list("ABCDEX")):
        ax_dict[ax_name].imshow(warped[i][0, 0, ...], cmap='gray')
        if ax_name == "A":
            ax_dict[ax_name].set_title("Affine")
        else:
            ax_dict[ax_name].set_title(f"Cascade {i}")

    ax_dict['L'].plot(train_loss, color='red', label='train_loss')
    ax_dict['L'].plot(val_loss, label='val_loss', color='blue')
    ax_dict['L'].plot(reg_loss, label='train_reg_loss', color='green')
    ax_dict['L'].set_title("Losses")
    ax_dict['L'].grid()
    ax_dict['L'].set_xlim(0, args.e)
    ax_dict['L'].legend(loc='upper right')
    ax_dict['L'].scatter(len(train_loss) - 1, train_loss[-1], s=20, color='red')
    ax_dict['L'].scatter(len(val_loss) - 1, val_loss[-1], s=20, color='blue')
    ax_dict['L'].scatter(len(reg_loss) - 1, reg_loss[-1], s=20, color='green')

    for i, ax_name in enumerate(list("GHIJKM")):
        plot_grid(ax_dict[ax_name], flows[i][0, ...])
        if ax_name == "G":
            ax_dict[ax_name].set_title("Affine")
        else:
            ax_dict[ax_name].set_title(f"Cascade {i}")

    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f'./ckp/visualization/epoch_{epoch}.png')