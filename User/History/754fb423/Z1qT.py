# visualize multivariate gaussian distribution using custom matrix


def plot_gaussian_3d(ax, mu, sigma, color="r", alpha=0.5):
    # create meshgrid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # create multivariate gaussian distribution
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mu, sigma)

    # plot the distribution
    ax.plot_surface(
        X,
        Y,
        rv.pdf(pos),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        alpha=alpha,
        color=color,
    )
