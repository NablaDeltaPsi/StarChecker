import copy, cv2, glob, sys, os, re, numpy as np, rawpy, warnings
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from scipy.stats import sigmaclip
from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
import tkinter as tk
import tkinter.font
import tkinter.messagebox # for pyinstaller
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter.filedialog import askopenfilename
from configparser import ConfigParser

warnings.simplefilter("ignore", category=OptimizeWarning)

GUINAME = "StarChecker"
VERSION = '1.0'

STAR_DETECTION_THRESHOLD = 0.4 # minimum brightness 0-1 between mean-clip and max
STAR_SELECTION_LIMIT = 0.9 # maximum brightness 0-1 between mean-clip and max (helps to skip burned stars)
STAR_WINDOW = 5 # size of the windows occupied by one star
MAX_STARS = 0 # limit stars to be fitted (0 is unlimited)
HIST_X = 'cnr' # histogram x-axis by parameter name ('ind', 'x', 'y', 'r', 'phi', 'amp', 'width', 'aspect')
HIST_C = 'amp' # histogram scatter color by parameter name ('ind', 'x', 'y', 'r', 'phi', 'amp', 'width', 'aspect')
LABEL_STARS = '' # label stars with parameter name ('', ind, 'x', 'y', 'r', 'phi', 'max', 'width', 'aspect')
PLOT_SINGLE_TYPE = '' # plot the fit of single stars ('guess', 'fit', 'error')
UPPERLIM_PLOT = 5

def process_file(
        file, # filename
        raw_auto_bright_thr=0.001, # percentage (0-1) of clipped values during autoscaling
        statistics_performance=2, # use only every nth pixel for statistics
        slopes=True,  # analyse image slopes
        analyze_stars=False, # detect and analyze stars in the image
        plot_result=True, # show image and plots of fit results
        region_size=300, # test region size, int or tuple or None (full image)
        region=0,  # corner from which the test region to take from (0 is center, -1 is edge)
        analyze_colors=1, # which color channels to be analyzed
        check_input=False, # show input image and exit
        return_colors_separately=True # return array instead of numbers with color-values
    ):
    total_timer = MyTimer()
    steps_timer = MyTimer()
    total_timer.start()

    # read
    print("\n\n" + "-"*50 + "\nFile:", file)
    steps_timer.start("read")
    image = read_image(file, raw_auto_bright_thr=raw_auto_bright_thr)
    steps_timer.stop()
    image_axes, color_axis = separate_axes(image)
    orig_imwidth = image.shape[image_axes[0]]
    orig_imheight = image.shape[image_axes[1]]

    # reduce
    if region_size is None:
        edge_size = None
    else:
        if not isinstance(region_size, (list, tuple, np.ndarray)):
            region_size = [region_size, region_size]
        if region >= 0:
            image, region_position = take_region(image, region_size, region=region)
            print("Reduced image to shape", image.shape)
            edge_size = None
        elif region == -1:
            edge_size = region_size[0]
            region_position = [0, 0]
        else:
            region_position = [0, 0]
            edge_size = None

    if check_input:
        fig = show_image(image)
        return None

    # statistics
    imformat = image_format(image)
    steps_timer.start("statistics")
    imstat = calc_image_statistics(image, clipped_statistics=True, every_nth=statistics_performance)
    steps_timer.stop()

    # slopes
    if slopes:
        steps_timer.start("slopes")
        imslopes = calc_color_slopes(image, every_nth=statistics_performance * 2)
        steps_timer.stop()
    else:
        imslopes = {}

    # print
    print_statistics(imformat | imstat | imslopes)

    if analyze_stars:
        # prepare for star analysis
        image = ensure_3d(image)
        image = order_axes(image)

        # which colors to be analyzed
        analyze_colors = ensure_list(analyze_colors)
        if analyze_colors is None or image.shape[2] == 1:
            analyze_colors = range(image.shape[2])
        analyze_colors = list(set(analyze_colors))

        # thresholds to be used
        this_star_det_threshold = np.add(imstat['mean-clip'], [STAR_DETECTION_THRESHOLD * x for x in np.subtract(imstat['max'], imstat['mean-clip'])]).astype('uint16')
        this_star_sel_limit = np.add(imstat['mean-clip'], [STAR_SELECTION_LIMIT * x for x in np.subtract(imstat['max'], imstat['mean-clip'])]).astype('uint16')
        print("   Detection thresholds:", this_star_det_threshold)
        print("   Selection limits:", this_star_sel_limit)

        # detect stars
        steps_timer.start("detect")
        stars = detect_stars(image, this_star_det_threshold, colors=analyze_colors, edge_size=edge_size)
        steps_timer.stop()
        print("   Stars detected (all colors):", myformat(len(stars)))

        # select stars
        stars = select_stars(stars, limits=this_star_sel_limit, maxstars=MAX_STARS)
        print("   Stars selected (all colors):", myformat(len(stars)))
        if not stars:
            print("No stars detected!")
            return None
        determine_star_positions(stars, [orig_imwidth, orig_imheight], region_position)

        # analyze stars
        steps_timer.start("analyze")
        stars, count_errors = fit_stars(image, stars)
        steps_timer.stop()
        print("   Fitted:", len(stars), "- Error:", count_errors)
        if not stars:
            print("Not a single successful fit!")
            return None

        # prepare plotting
        histx   = get_attributes_by_color(stars, HIST_X, ignore_color=not return_colors_separately)
        histc   = get_attributes_by_color(stars, HIST_C, ignore_color=not return_colors_separately)
        widths  = get_attributes_by_color(stars, 'width', ignore_color=not return_colors_separately)
        aspects = get_attributes_by_color(stars, 'aspect', ignore_color=not return_colors_separately)
        amps    = get_attributes_by_color(stars, 'amp', ignore_color=not return_colors_separately)
        mean_widths  = [np.mean(sigmaclip(x, low=2, high=2)[0]) for x in widths]
        mean_aspects = [np.mean(sigmaclip(x, low=2, high=2)[0]) for x in aspects]
        mean_width = np.nanmean(mean_widths)
        mean_aspect = np.nanmean(mean_aspects)
        print("   Average star FWHM:", myformat(mean_width))
        print("   Average star aspect ratio:", myformat(mean_aspect))

        print("Total time for file:", end=' ')
        total_timer.stop()

        # plot
        if plot_result:
            plt_image = highlight_stars(image, stars, analyze_colors)
            show_image(plt_image, stars=stars, colors=analyze_colors)
            show_scatter(histx, widths, histc, colors=analyze_colors, means=mean_widths, figure_name='FWHM', x_name='FWHM')
            show_scatter(histx, aspects, histc, colors=analyze_colors, means=mean_aspects, figure_name="Aspect ratio", x_name="Aspect ratio")

        return_stars = [len(x) for x in histx]
        return_widths = mean_widths
        return_aspects = mean_aspects
        print("RESULT:", return_stars, return_widths, return_aspects)
        return return_stars, return_widths, return_aspects




# =================================================
# MAIN ROUTINES
# =================================================

def read_image(file, ensure_3d=True, raw_auto_bright_thr=0.001):
    if not os.path.isfile(file):
        print("File does not exist!\n\n")
        sys.exit()

    filetype = file.split(".")[-1].lower()
    if filetype in ['fit', 'fits']:
        image = fits.getdata(file, ext=0)
    elif filetype in ['arw']:
        image = rawpy.imread(file).postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm(3), # 3=AHD, 4=DCB
            use_auto_wb=True,
            auto_bright_thr=raw_auto_bright_thr,
            output_bps=16,
        )
    else:
        image = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED) # cv2.IMREAD_ANYDEPTH, cv2.IMREAD_UNCHANGED
    if image is None:
        print("Image is None, probably unable to read it!\n\n")
        sys.exit()
    return image

def take_region(image, size, region=0):
    print(image.shape, region, size)
    if region == 5:
        region = 0
    position = []
    for i in [0,1]:
        image_axis_length = image.shape[i]
        if region == 0:
            range_from = image_axis_length / 2 - size[i] / 2
            range_to   = image_axis_length / 2 + size[i] / 2
        elif region == 1 or (region == 2 and i == 0) or (region == 3 and i == 1):
            range_from = 0
            range_to = size[i]
        elif region == 4 or (region == 2 and i == 1) or (region == 3 and i == 0):
            range_from = image_axis_length - size[i]
            range_to   = image_axis_length
        else:
            raise Exception("Corner should be between 0 and 4!")
        # safety
        range_from = int(max(0,range_from))
        range_to = int(min(range_to, image_axis_length))
        position.append(range_from)
        image = np.take(image, range(range_from, range_to), i)
    return image, position

def calc_image_statistics(image_, clipped_statistics=False, every_nth=0):
    image = copy.deepcopy(image_)
    if every_nth:
        if len(image.shape) == 2:
            for i in range(2):
                thisrange  = list(range(0, image.shape[i], 2 * every_nth))
                thisrange += list(range(1, image.shape[i], 2 * every_nth))
                thisrange.sort()
                image = np.take(image, thisrange, i)
        else:
            image = image[::every_nth,::every_nth,:]
    image_axes, color_axis = separate_axes(image)
    if clipped_statistics:
        clippedvals = calc_clippedvals(image, sigma=3)
    images_2d = [image]
    if color_axis:
        for c in range(image.shape[color_axis]):
            images_2d.append(np.take(image, c, color_axis))
            clippedvals.append(calc_clippedvals(np.take(image, c, color_axis), sigma=3))
    else:
        for i in range(4):
            images_2d.append(get_bayer(image, i))
            clippedvals.append(calc_clippedvals(images_2d[-1], sigma=3))
    imstat = {}
    imstat['median'] = [np.median(x) for x in images_2d]
    imstat['mean'] = [np.mean(x) for x in images_2d]
    if clipped_statistics:
        imstat['mean-clip'] = [clipstat(x, 'mean', sigma=3) for x in clippedvals]
    imstat['stdv'] = [np.std(x) for x in images_2d]
    if clipped_statistics:
        imstat['stdv-clip'] = [clipstat(x, 'std', sigma=3) for x in clippedvals]
    imstat['min'] = [np.min(x) for x in images_2d]
    imstat['max'] = [np.max(x) for x in images_2d]
    imstat['nans'] = [np.count_nonzero(np.isnan(x)) for x in images_2d]
    if clipped_statistics:
        imstat['uniques'] = [clipstat(x, 'unique', sigma=-1) for x in clippedvals]
        imstat['uniques-clip'] = [clipstat(x, 'unique', sigma=3) for x in clippedvals]
    imstat['minstep'] = [np.min(np.diff(sorted(np.unique(x)))) for x in images_2d]
    imstat['maxsteps'] = np.divide(imstat['max'], imstat['minstep'])
    imstat['minbit'] = np.log2(imstat['maxsteps'])
    imstat['median (max)'] = 100 * np.divide(imstat['median'], imstat['max'])
    imstat['stdv (median)'] = 100 * np.divide(imstat['stdv'], imstat['median'])
    return imstat

def calc_slope(image_, color_index=0, every_nth=8, ignore_region=10):
    image = copy.deepcopy(image_)
    image_axes, color_axis = separate_axes(image)
    if color_axis:
        image = image.take(indices=color_index, axis=color_axis)
    if every_nth:
        image = image[ignore_region:-ignore_region:every_nth, ignore_region:-ignore_region:every_nth]
        ignore_region = int(ignore_region / every_nth) + 1
    height = image.shape[0]
    width = image.shape[1]
    rowmeans = []
    for row in range(height):
        rowmeans.append(np.mean(sigmaclip(image[row, :], low=2, high=2)[0]))
    slope_y = np.mean(np.diff(rowmeans) / every_nth) * image_.shape[0]

    colmeans = []
    for col in range(width):
        colmeans.append(np.mean(sigmaclip(image[:, col], low=2, high=2)[0]))
    slope_x = np.mean(np.diff(colmeans) / every_nth) * image_.shape[1]
    return slope_x, slope_y

def calc_color_slopes(image, every_nth=8):
    keylen = 20
    vallen = 9
    image_axes, color_axis = separate_axes(image)
    imslopes = {}
    imslopes['slopes_x'] = [0]
    imslopes['slopes_y'] = [0]
    if color_axis:
        for c in range(image.shape[color_axis]):
            slope_x, slope_y = calc_slope(image, color_index=c, every_nth=every_nth)
            imslopes['slopes_x'].append(100 * slope_x / np.median(np.take(image, c, 2)))
            imslopes['slopes_y'].append(100 * slope_y / np.median(np.take(image, c, 2)))
    return imslopes

def detect_stars(image, threshold, colors=None, remove_overlapping=True, neighbor_threshold=True, edge_size=False):
    threshold = ensure_list(threshold)
    colors = ensure_list(colors)
    if len(threshold) < image.shape[2]:
        threshold = threshold[0] * np.ones(image.shape[2])
    else:
        threshold = threshold[-image.shape[2]:]
    if colors is None:
        colors = range(image.shape[2])
    if len(colors) == 1 and colors[0] == 1 and image.shape[2] == 1:
        colors = [0]
    occupation_array = np.zeros(image.shape).astype(int)
    overlap_star_indices = []
    imheight = image.shape[0]
    imwidth = image.shape[1]
    stars = []
    for c in colors:
        thisimage = np.take(image, c, 2)
        for row in range(STAR_WINDOW + 1, imheight - STAR_WINDOW - 1):
            for col in range(STAR_WINDOW + 1, imwidth - STAR_WINDOW - 1):
                if edge_size:
                    if not min([row, imheight-row, col, imwidth-col]) < edge_size:
                        continue

                thisval = thisimage[row, col]

                # check threshold
                if thisval < threshold[c]:
                    continue

                # check marker
                if occupation_array[row, col, c] > 0:
                    continue

                # check nearest neighbors
                neighbor_continue = False
                neighbor_sum = 0
                for neighbor in [[row-1, col], [row+1, col], [row, col-1], [row, col+1]]:
                    neighbor_val = thisimage[neighbor[0], neighbor[1]]
                    neighbor_sum += neighbor_val
                    if thisval < neighbor_val: # thisval is no maximum
                        neighbor_continue = True
                        break
                if neighbor_continue:
                    continue
                if neighbor_threshold:
                    if neighbor_sum < 0.6 * 4 * thisval:  # thisval is hotpixel, not a star
                        continue

                # check complete STAR_WINDOW
                if np.max(thisimage[row-STAR_WINDOW:row+STAR_WINDOW+1, col-STAR_WINDOW:col+STAR_WINDOW+1]) > thisval:
                    continue

                # no continue until here -> found a star
                stars.append({
                    'ind': len(stars),
                    'x': row,
                    'y': col,
                    'color': c,
                    'max': thisval
                })

                # mark star area
                for i in range(row - STAR_WINDOW, row + STAR_WINDOW + 1):
                    for j in range(col - STAR_WINDOW, col + STAR_WINDOW + 1):

                        # check already occupied, then set as overlapping and remember both star indices
                        if remove_overlapping:
                            this_occupation_marker = occupation_array[i,j,c]
                            if this_occupation_marker > 0:
                                if this_occupation_marker not in overlap_star_indices:
                                    overlap_star_indices.append(occupation_array[i,j,c])
                                if stars[-1]['ind'] not in overlap_star_indices:
                                    overlap_star_indices.append(stars[-1]['ind'])

                        # always mark with current star
                        occupation_array[i,j,c] = stars[-1]['ind']

    if remove_overlapping:
        stars_out = []
        for i in range(len(stars)):
            if i not in overlap_star_indices:
                stars_out.append(stars[i])
    else:
        stars_out = stars
    return stars_out

def determine_star_positions(stars, image_shape, region_position):
    corner_phi = []
    corner_phi.append(np.rad2deg(np.arctan2(image_shape[0], image_shape[1])))
    corner_phi.append(np.rad2deg(np.arctan2(image_shape[0], -image_shape[1])))
    corner_phi.append(np.rad2deg(np.arctan2(-image_shape[0], -image_shape[1])))
    corner_phi.append(np.rad2deg(np.arctan2(-image_shape[0], image_shape[1])))
    for i in range(len(corner_phi)):
        if corner_phi[i] < 0:
            corner_phi[i] += 360
    for star in stars:
        true_x = star['x'] + region_position[0]
        true_y = star['y'] + region_position[1]
        rel_x = true_x - image_shape[0] / 2
        rel_y = true_y - image_shape[1] / 2
        star['r'] = np.sqrt(abs(rel_x) ** 2 + abs(rel_y) ** 2)
        star['phi'] = np.rad2deg(np.arctan2(-rel_x, rel_y))
        if star['phi'] < 0:
            star['phi'] += 360
        if star['phi'] <= corner_phi[0]:
            star['cnr'] = 3.5 + 0.5 * star['phi'] / corner_phi[0]
        elif star['phi'] <= corner_phi[1]:
            star['cnr'] = 0 + (star['phi'] - corner_phi[0]) / (corner_phi[1] - corner_phi[0])
        elif star['phi'] <= corner_phi[2]:
            star['cnr'] = 1 + (star['phi'] - corner_phi[1]) / (corner_phi[2] - corner_phi[1])
        elif star['phi'] <= corner_phi[3]:
            star['cnr'] = 2 + (star['phi'] - corner_phi[2]) / (corner_phi[3] - corner_phi[2])
        else:
            star['cnr'] = 3 + 0.5 * (star['phi'] - corner_phi[3]) / (360 - corner_phi[3])

def select_stars(stars, limits=[], maxstars=1000):
    stars_out = []
    for star in sorted(stars, key=lambda x: x['max'], reverse=False):
        if not limits.any():
            stars_out.append(star)
        elif star['max'] <= limits[star['color']]:
            stars_out.append(star)
        if maxstars > 0 and len(stars_out) >= maxstars:
            break
    stars_out.sort(key=lambda x: x['max'], reverse=True)
    return stars_out

def highlight_stars(image_, stars, colors):
    image = copy.deepcopy(image_)
    if image.shape[2] == 1:
        colors = 0
    colors = ensure_list(colors)
    if colors is None or len(colors) == image.shape[2] > 1:
        colors = range(image.shape[2])
    else:
        image = np.mean(image[:,:,colors], axis=2)
        image = np.array([image, image, image])
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)
    for color in colors:
        if image.shape[2] > 1:
            highlight_color = get_highlight_color(color, np.max(image), 0.7)
        else:
            highlight_color = np.max(image)
        limx = image.shape[0]
        limy = image.shape[1]
        for star in stars:
            if not star["color"] == color:
                continue
            wl = star['x']-STAR_WINDOW-1
            wr = star['x']+STAR_WINDOW+1
            wu = star['y']-STAR_WINDOW-1
            wb = star['y']+STAR_WINDOW+1
            if not (wl < 0 or wr >= limx or wu < 0 or wb >= limy):
                image[wl:wr+1, wu, :] = highlight_color
                image[wl:wr+1, wb, :] = highlight_color
                image[wl, wu:wb+1, :] = highlight_color
                image[wr, wu:wb+1, :] = highlight_color
    return image

def fit_stars(image, stars):
    count_errors = 0
    new_stars = []
    for star in stars:
        this_region = np.take(image,       range(star['x']-STAR_WINDOW, star['x']+STAR_WINDOW+1), 0)
        this_region = np.take(this_region, range(star['y']-STAR_WINDOW, star['y']+STAR_WINDOW+1), 1)
        this_region = np.take(this_region, star['color'], 2)
        res = analyze_region(this_region, starinfo=star)
        if res:
            amp, width, aspect = res
        else:
            count_errors += 1
            continue
        width = 2 * np.sqrt(2 * np.log(2)) * width # FWHM
        new_stars.append(star)
        new_stars[-1]['amp'] = amp
        new_stars[-1]['width'] = width
        new_stars[-1]['aspect'] = aspect
    return new_stars, count_errors

def analyze_region(region, plot_type=PLOT_SINGLE_TYPE, starinfo=None):
    if not region.shape[0] == region.shape[1]:
        raise Exception("Region must be square!")
    region_size = region.shape[0]
    x = np.arange(region_size)
    y = np.arange(region_size)
    x, y = np.meshgrid(x, y)
    initial_guess = (
        np.max(region),
        (region_size-1)/2,
        (region_size-1)/2,
        region_size / 6,
        region_size / 6,
        np.deg2rad(0),
        np.min(region)
    )
    if plot_type == 'guess':
        fig, ax = plt.subplots(1, 1)
        ax.imshow(region.reshape(region_size, region_size), cmap='gray', origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        data_guessed = twoD_Gaussian((x, y), *initial_guess)
        ax.contour(x, y, data_guessed.reshape(region_size, region_size), 4, colors='r')
        plt.show()
    try:
        popt, _ = curve_fit(twoD_Gaussian, (x, y), region.ravel(), p0=initial_guess)
    except:
        if plot_type == 'error':
            if starinfo:
                print(starinfo)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(region.reshape(region_size, region_size), cmap='gray', origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
            data_guessed = twoD_Gaussian((x, y), *initial_guess)
            ax.contour(x, y, data_guessed.reshape(region_size, region_size), 4, colors='r')
            plt.show()
        return None
    if plot_type == 'fit':
        fig, ax = plt.subplots(1, 1)
        ax.imshow(region.reshape(region_size, region_size), cmap='gray', origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        data_fitted = twoD_Gaussian((x, y), *popt)
        ax.contour(x, y, data_fitted.reshape(region_size, region_size), 4, colors='r')
        plt.show()
    amp = popt[0]
    widths = [popt[3], popt[4]]
    widths.sort()
    aspect = widths[1] / widths[0]
    if not (
        np.min(region) < amp < np.max(region)*1.5 and \
        0 < widths[0] < region_size/2 and \
        1 <= aspect < 10):
        if plot_type == 'error':
            if starinfo:
                print(starinfo)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(region.reshape(region_size, region_size), cmap='gray', origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
            data_guessed = twoD_Gaussian((x, y), *initial_guess)
            ax.contour(x, y, data_guessed.reshape(region_size, region_size), 4, colors='r')
            plt.show()
        return None
    return amp, widths[0], aspect







# =====================================================
# SMALL HELPER FUNCTIONS
# =====================================================

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset, ravel=True):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    if ravel:
        return g.ravel()
    else:
        return g

def to_printdict(origdict):
    printdict = {}
    for key in origdict.keys():
        printdict[key] = []
        perc = True if key.find("(") >= 0 or key.find("Slope") >= 0 else False
        if isinstance(origdict[key], (list, np.ndarray)):
            for i in range(len(origdict[key])):
                printdict[key].append(myformat(origdict[key][i], perc=perc))
        else:
            printdict[key].append(myformat(origdict[key], perc=perc))
    return printdict

def print_statistics(origdict):
    printdict = to_printdict(origdict)
    keylen  = 0
    vallens = [0] * max([len(val) for key, val in printdict.items()])
    for key, val in printdict.items():
        keylen = max(keylen, len(key))
        for i in range(len(vallens)):
            if len(val) > i:
                vallens[i] = max(vallens[i], len(val[i]))
    print("\n" + "-"*60)
    for key, val in printdict.items():
        print(key.ljust(keylen), "  ".join([x.rjust(vallens[ind]) for ind, x in enumerate(val)]))
    print("-"*60 + "\n")

def calc_clippedvals(image, sigma, colors_separate=False):
    clippedvals = []
    if sigma == -1:
        clippedvals.append(image.flatten())
    else:
        clippedvals.append(sigmaclip(image, low=sigma, high=sigma)[0])
    if colors_separate and len(image.shape) > 2:
        for c in range(image.shape[2]):
            thisimage = np.take(image, c, 2)
            if sigma == -1:
                clippedvals.append(thisimage.flatten())
            else:
                clippedvals.append(sigmaclip(thisimage, low=sigma, high=sigma)[0])
    return clippedvals

def clipstat(vals, stattype, sigma=3):
    if not isinstance(vals, list): # image instead of list of clippedvals
        clippedvals = calc_clippedvals(vals, sigma)
    else:
        clippedvals = vals # precalculated
    clipstat = []
    for i in range(len(clippedvals)):
        if stattype == 'std':
            clipstat.append(np.std(clippedvals[i]))
        elif stattype == 'mean':
            clipstat.append(np.mean(clippedvals[i]))
        elif stattype == 'count':
            clipstat.append(np.count_nonzero(clippedvals[i]))
        elif stattype == 'unique':
            clipstat.append(len(np.unique(clippedvals[i])))
        else:
            raise Exception("stattype " + stattype + " does not exist!")
    if len(clipstat) == 1:
        clipstat = clipstat[0]
    return clipstat

def myformat(x, perc=False, precision=3, thousand_sep=True):
    if x is None:
        return ""
    if isinstance(x, (tuple, str)):
        str_out = str(x)
    elif isinstance(x, int) or abs(x) > 120:
        if not isinstance(x, int):
            x = int(x)
        if abs(x) > 99999:
            if thousand_sep:
                str_out = '{:,}'.format(x).replace(',', '.')
            else:
                str_out = '{:,}'.format(x).replace(',', '')
        else:
            str_out = str(x)
    else:
        format_string = "{:#." + str(precision) + "g}"
        str_out = format_string.format(x)
    if perc:
        str_out += "%"
    return str_out

def separate_axes(image):
    # try to find a rgb dimension
    color_axis = None
    for d in range(len(image.shape)):
        if image.shape[d] < 5:
            color_axis = d
            break

    # calc tuple of image_axes
    image_axes = []
    for d in range(len(image.shape)):
        if not d == color_axis:
            image_axes.append(d)
    image_axes = tuple(image_axes)

    return image_axes, color_axis

def new_tkinter_graph(fig, title='Graph window', square=False):
    this_tk = tk.Tk()
    this_tk.title(title)
    if square:
        this_tk.geometry('500x500+200+100')
    else:
        this_tk.geometry('700x500+200+100')
    try:
        this_tk.iconbitmap(GUINAME + '.ico')
    except:
        print("Did not find icon... continuing...")
    canvas = FigureCanvasTkAgg(fig, this_tk)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.01, rely=0.01, relheight=0.98, relwidth=0.98, anchor='nw')
    toolbar = NavigationToolbar2Tk(canvas)
    toolbar.update()
    canvas.get_tk_widget().pack()

def show_image(image, gamma=0, stars=[], colors=[]):
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    plt.get_current_fig_manager().set_window_title("Star image")
    if gamma > 0:
        image = apply_gamma(image, gamma)
    image = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    if LABEL_STARS:
        for i in range(len(stars)):
            if stars[i]['color'] in colors:
                thislabel = "   " + myformat(stars[i][LABEL_STARS])
                ax.text(stars[i]['y'], stars[i]['x'], thislabel, color=get_highlight_color(2 - stars[i]['color'], 1, 0.7), va='center', ha='left', fontsize=6)
    new_tkinter_graph(fig, title="Star image", square=True)

def show_scatter(x, y, z, colors=[], means=None, figure_name='', x_name=''):
    if not colors:
        colors = range(len(x))
    colors = ensure_list(colors)
    fig, axs = plt.subplots(len(colors), 1)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    axs = ensure_list(axs)
    if figure_name:
        plt.get_current_fig_manager().set_window_title(figure_name)
    cmaps = ['Blues', 'Greens', 'Reds']
    for i in range(len(colors)):
        c = colors[i]
        thiscmap = cmaps[c] if c < len(cmaps) else 'binary'
        axs[i].scatter(x[i], y[i], c=z[i], s=4, marker='o', cmap=thiscmap, vmin=0, vmax=1.2 * max(z[i]))
        if means:
            axs[i].plot([0, max([max(x) for x in x if x] + [4])], [means[i], means[i]], color='black')
            axs[i].text(0, means[i], myformat(means[i]), va='bottom', color='black')
        axs[i].set_xlim([0, max([max(x) for x in x if x] + [4])])
        axs[i].set_ylim([0, UPPERLIM_PLOT])
        if i == 0 and x_name:
            axs[i].set_title(x_name)
        if not i == len(colors) - 1:
            axs[i].set_xticklabels([])
        if i == len(colors) - 1:
            if HIST_X == 'phi':
                this_xlabel = 'Angle (0 to right, counter-clockwise)'
            elif HIST_X == 'cnr':
                this_xlabel = 'Corner (counter-clockwise from top-right)'
                axs[i].set_xticks([0,1,2,3,4])
                axs[i].set_xticklabels(['TR','TL','BL','BR','TR'])
            elif HIST_X == 'amp':
                this_xlabel = 'Amplitude'
            else:
                this_xlabel = HIST_X
            axs[i].set_xlabel(this_xlabel)
    new_tkinter_graph(fig, title=figure_name)

def ensure_3d(image_):
    if len(image_.shape) == 2:
        image = copy.deepcopy(image_)
        image = image[:,:,np.newaxis]
    else:
        image = image_
    return image

def ensure_list(input):
    if input is None:
        return None
    elif isinstance(input, (tuple, list, range, np.ndarray)):
        return input
    else:
        return [input]

def image_format(image):
    image_axes, color_axis = separate_axes(image)
    imformat = {}
    imformat['shape'] = image.shape
    imformat['color axis'] = color_axis
    imformat['image axes'] = image_axes
    imformat['count'] = [np.prod(image.shape)]
    if color_axis:
        for c in range(image.shape[color_axis]):
            imformat['count'].append(np.prod(np.take(image, c, color_axis).shape))
    return imformat

def get_bayer_indices(index):
    # for consistency
    index_mapping = [[0, 0], [0, 1], [1, 0], [1, 1]]
    return index_mapping[index]

def get_bayer(image_, index):
    image = copy.deepcopy(image_)
    i, j = get_bayer_indices(index)
    range_0 = range(i, image.shape[0], 2)
    range_1 = range(j, image.shape[1], 2)
    image = np.take(image, range_0, 0)
    image = np.take(image, range_1, 1)
    return image

def order_axes(image, type='HWC'):
    image_shape = image.shape
    axes_sort = np.argsort(image_shape)[::-1]
    image = np.transpose(image, axes=axes_sort)
    if type == 'HWC':
        image = np.swapaxes(image, 0, 1)
    if len(image.shape) == 3 and type == 'CHW':
        image = np.swapaxes(image, 0, 2)
    return image

def get_highlight_color(color, maxval, brightness):
    brightness = brightness * maxval
    if color == 0:
        return np.array([maxval, brightness, brightness])
    elif color == 1:
        return np.array([brightness, maxval, brightness])
    else:
        return np.array([brightness, brightness, maxval])

def cmd_param_to_kwargs(cmd_params, keys_allowed=[], prefix='-', split='='):
    kwargs = {}
    for i in range(len(cmd_params)):
        cmd_split = cmd_params[i].strip().split(split)
        if not len(cmd_split) == 2:
            raise Exception("Parameters should start with '" + prefix + "' and contain exactly one '" + split + "'!")
        key = cmd_split[0]
        value = cmd_split[1]
        if not key[:len(prefix)] == prefix:
            raise Exception("Parameters should start with '" + prefix + "' and contain exactly one '" + split + "'!")
        key = key[len(prefix):]
        if keys_allowed and key not in keys_allowed:
            raise Exception("Keyword '" + key + "' unknown. Allowed keywords: '" + "', '".join(keys_allowed) + "'.")
        kwargs[key] = eval(value, {}, {})
    return kwargs

def get_attributes_by_color(stars, key, ignore_color=False):
    attdict = {}
    for star in stars:
        if star['color'] not in attdict.keys():
            attdict[star['color']] = []
        attdict[star['color']].append(star[key])
    attlist = []
    for key in sorted(attdict.keys()):
        if ignore_color:
            attlist += attdict[key]
        else:
            attlist.append(attdict[key])
    if ignore_color:
        attlist = [attlist]
    return attlist

def apply_gamma(image, gamma):
    image = np.subtract(image, np.min(image))
    maxval = np.max(image)
    image = np.multiply(np.power(np.divide(image, maxval), 1/gamma), maxval)
    return image.astype('uint16')

def contains(string, substrings):
    if not isinstance(substrings, list):
        substrings = [substrings]
    for sub in substrings:
        if string.lower().find(sub.lower()) >= 0:
            return True
    return False

def rgbtohex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

def get_number(input_string, default=None):
    output = ''
    for i in range(len(input_string)):
        if input_string[i].isnumeric():
            output += input_string[i]
    if len(output) == 0:
        if default is not None:
            output = default
        else:
            output = None
    else:
        output = int(output)
    return output

def csv_sep_line(array):
    flattened = []
    for x in array:
        if isinstance(x, (list, tuple, np.ndarray)):
            for y in x:
                flattened.append(y)
        else:
            flattened.append(x)
    return "\"" + "\";\"".join([myformat(x, precision=4, thousand_sep=False) for x in flattened]) + "\"\n"

def abab_array(list_of_arrays):
    array_lengths = [len(x) for x in list_of_arrays]
    if not len(set(array_lengths)) <= 1:
        raise Exception("Arrays must have same length!")
    output = []
    for i in range(array_lengths[0]):
        for array in list_of_arrays:
            output.append(array[i])
    return output

class MyTimer():
    def __init__(self):
        self.timertime = datetime.now()
    def start(self, *args):
        if len(args) > 0:
            print(args[0] + "...", end=' ')
        self.timertime = datetime.now()
    def stop(self):
        time_s = (datetime.now() - self.timertime).total_seconds()
        if time_s >= 120:
            print("<" + myformat(time_s/60) + "min>")
        else:
            print("<" + myformat(time_s) + "s>")
        self.timertime = datetime.now()

class NewGUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUINAME + " v" + VERSION)
        self.lastpath = ""
        self.loaded_files = []

        self.region = 0
        self.region_size = 300
        self.plot_result = True
        self.analyze_colors = 1
        self.return_colors_separately = False

        self.root.protocol("WM_DELETE_WINDOW",  self.on_close)

        # icon and DPI
        try:
            self.root.iconbitmap(GUINAME + ".ico")
            self.root.update() # important: recalculate the window dimensions
        except:
            print("Found no icon.")

        # menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # region
        menu_region = tk.Menu(menubar, tearoff=0)
        self.radio_region = tk.StringVar(self.root)
        options = ["edge", "center", "corner (TL)", "corner (TR)", "corner (BR)", "corner (BL)"]
        for opt in options:
            menu_region.add_radiobutton(label=opt, value=opt, variable=self.radio_region)
        menubar.add_cascade(label="Region", menu=menu_region)

        # size
        menu_size = tk.Menu(menubar, tearoff=0)
        self.radio_size = tk.StringVar(self.root)
        options = ["100","200","300","400","500","1000","full"]
        for opt in options:
            menu_size.add_radiobutton(label=opt, value=opt, variable=self.radio_size)
        menubar.add_cascade(label="Size", menu=menu_size)

        # colors
        menu_color = tk.Menu(menubar, tearoff=0)
        self.radio_color = tk.StringVar(self.root)
        options = ["All","0","1","2"]
        for opt in options:
            menu_color.add_radiobutton(label=opt, value=opt, variable=self.radio_color)
        menubar.add_cascade(label="Color", menu=menu_color)

        # # options
        # anzeige = tk.Menu(menubar, tearoff=0)
        # self.check_option_1 = tk.BooleanVar()
        # self.check_option_2 = tk.BooleanVar()
        # anzeige.add_checkbutton(label="Option 1", onvalue=1, offvalue=0, variable=self.check_option_1)
        # anzeige.add_checkbutton(label="Option 2", onvalue=1, offvalue=0, variable=self.check_option_2)
        # menubar.add_cascade(label="Options", menu=anzeige)

        # buttons and labels
        self.button_load = tk.Button(text="Load files", command=self.load_files)
        self.button_load.grid(row=0, column=0, sticky='NWSE', padx=10, pady=5)
        self.label_files_var = tk.StringVar()
        self.label_files = tk.Label(textvariable=self.label_files_var, justify='left')
        self.label_files.grid(row=1, column=0, sticky='NWSE', padx=10, pady=5)
        self.update_labels()
        self.button_show = tk.Button(text="Show", command=self.process_show)
        self.button_show.grid(row=2, column=0, sticky='NWSE', padx=10, pady=5)
        self.button_start = tk.Button(text="Start", command=self.process_start)
        self.button_start.grid(row=3, column=0, sticky='NWSE', padx=10, pady=5)

        # configure
        self.root.grid_columnconfigure(0,weight=1)
        self.root.grid_rowconfigure(0,weight=1)
        self.root.grid_rowconfigure(1,weight=1)
        self.root.grid_rowconfigure(2,weight=1)
        self.root.grid_rowconfigure(3,weight=1)

        # default configs
        self.load_config_file()

        # mainloop
        self.root.mainloop()

    def on_close(self):
        print("... save config file")
        config_object = ConfigParser()
        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["window size"] = self.root.winfo_geometry()
        config_object["SETTINGS"]["lastpath"] = self.lastpath
        config_object["SETTINGS"]["files"] = ",".join(self.loaded_files)
        config_object["SETTINGS"]["region"] = self.radio_region.get()
        config_object["SETTINGS"]["size"] = self.radio_size.get()
        config_object["SETTINGS"]["color"] = self.radio_color.get()

        with open(GUINAME + ".conf", 'w') as conf:
            config_object.write(conf)

        self.root.quit()
        self.root.destroy()

    def load_config_file(self):
        try:
            config_object = ConfigParser()
            if os.path.exists(GUINAME + ".conf"):
                config_object.read(GUINAME + ".conf")
                print("load config file")
            else:
                print("use default settings")
                config_object["SETTINGS"] = {
                    "window size":  '278x169+198+135',
                    "lastpath":     './',
                    "files":       '',
                    "region":       'edge',
                    "size":       '300',
                    "color":        '1',
                }

            # apply
            self.root.geometry(config_object["SETTINGS"]["window size"])
            self.lastpath = config_object["SETTINGS"]["lastpath"]
            self.loaded_files = [x for x in config_object["SETTINGS"]["files"].split(',') if x]
            self.radio_region.set(config_object["SETTINGS"]["region"])
            self.radio_size.set(config_object["SETTINGS"]["size"])
            self.radio_color.set(config_object["SETTINGS"]["color"])

            self.update_labels()

        except:
            print("Unable to load config file!")

    def load_files(self):
        self.loaded_files = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('all', '.*'), ('.arw', '.arw')])
        self.lastpath = os.path.dirname(self.loaded_files[0])
        self.update_labels()
        return

    def update_labels(self, status=""):
        if status:
            self.label_files_var.set(status)
            if contains(self.label_files_var.get().lower(), ["ready", "finish"]):
                self.label_files.configure(background=rgbtohex(180, 230, 180))
            elif contains(self.label_files_var.get().lower(), ["error", "interr", "stop", "no file"]):
                self.label_files.configure(background=rgbtohex(250, 180, 180))
            elif contains(self.label_files_var.get().lower(), ["write", "."]):
                self.label_files.configure(background=rgbtohex(240, 230, 180))
            else:
                self.label_files.configure(background=rgbtohex(240, 210, 180))
        else:
            self.label_files_var.set(str(len(self.loaded_files)) + " files")
            if len(self.loaded_files) > 0:
                self.label_files.configure(background=rgbtohex(210, 230, 255))
            else:
                self.label_files.configure(background=rgbtohex(250, 180, 180))
        self.root.update()
        return

    def set_parameters(self):
        # determine region
        if contains(self.radio_region.get(), "edge"):
            self.region = -1
        elif contains(self.radio_region.get(), "center"):
            self.region = 0
        elif contains(self.radio_region.get(), "(TL)"):
            self.region = 1
        elif contains(self.radio_region.get(), "(TR)"):
            self.region = 2
        elif contains(self.radio_region.get(), "(BL)"):
            self.region = 3
        elif contains(self.radio_region.get(), "(BR)"):
            self.region = 4
        else:
            self.region = 0
        if contains(self.radio_size.get(), "full"):
            self.region_size = None
        else:
            self.region_size = (int(self.radio_size.get()), int(self.radio_size.get()))
        self.plot_result = True if len(self.loaded_files) == 1 else False
        self.analyze_colors = [0,1,2] if self.radio_color.get() == 'All' else int(self.radio_color.get())
        self.analyze_colors = ensure_list(self.analyze_colors)
        if len(self.loaded_files) > 1:
            self.return_colors_separately = False
        else:
            self.return_colors_separately = True

    def process_show(self):
        try:
            if len(self.loaded_files) == 0:
                self.update_labels("no file!")
                return
            self.update_labels(status=os.path.basename(self.loaded_files[0]))
            self.set_parameters()
            process_file(self.loaded_files[0], region_size=self.region_size, region=self.region, check_input=True)
        except Exception as e:
            tk.messagebox.showerror('Error', repr(e))
            raise e
        return

    def process_start(self):
        try:
            if len(self.loaded_files) == 0:
                self.update_labels("no file!")
                return
            self.set_parameters()

            files_nr = []
            files_imagenr = []
            files_stars = []
            files_widths = []
            files_aspects = []

            for i in range(len(self.loaded_files)):
                file = self.loaded_files[i]
                self.update_labels(status=os.path.basename(file))
                res = process_file(
                    self.loaded_files[i],
                    statistics_performance=4,
                    slopes=False,
                    analyze_stars=True,
                    region_size=self.region_size,
                    region=self.region,
                    analyze_colors=self.analyze_colors,
                    check_input=False,
                    plot_result=self.plot_result,
                    return_colors_separately=self.return_colors_separately,
                )
                if res is None:
                    continue
                files_stars.append(res[0])
                files_widths.append(res[1])
                files_aspects.append(res[2])
                files_nr.append(i + 1)
                files_imagenr.append(get_number(os.path.basename(self.loaded_files[i]), default=0))

            self.update_labels(status="save")
            with open(GUINAME + ".csv", 'w') as f:
                header = ["FILE", "IMAGE"]
                if len(files_stars[0]) > 1:
                    for i in range(len(files_stars[0])):
                        header += [x+"_"+str(i) for x in ["STARS", "FWHM", "ASPECT"]]
                else:
                    header += ["STARS", "FWHM", "ASPECT"]
                f.write(csv_sep_line(header))
                for i in range(len(files_stars)):
                    file_abab_array = abab_array([files_stars[i], files_widths[i], files_aspects[i]])
                    f.write(csv_sep_line([files_nr[i]] + [files_imagenr[i]] + file_abab_array))
            print("Saved data as CSV.")

            if len(self.loaded_files) > 1:
                fig, axs = plt.subplots()
                fig.set_figheight(20)
                fig.set_figwidth(20)
                plt.get_current_fig_manager().set_window_title(GUINAME)
                labels = files_imagenr if 0 not in files_imagenr else files_nr
                colors = ['blue', 'green', 'red']
                for c in range(len(files_stars[0])):
                    axs.scatter([x[c] for x in files_aspects], [x[c] for x in files_widths], s=20, color=colors[self.analyze_colors[c]], marker='o')
                for i in range(len(labels)):
                    axs.text(files_aspects[i][0], files_widths[i][0], '   '+str(labels[i]), va='center', ha='left', fontsize=8)
                axs.set_xlabel('Aspect ratio')
                axs.set_ylabel('FWHM')
                axs.set_xlim([0, UPPERLIM_PLOT])
                axs.set_ylim([0, UPPERLIM_PLOT])
                new_tkinter_graph(fig, title='Overview')
            self.update_labels("finished.")

        except Exception as e:
            tk.messagebox.showerror('Error', repr(e))
            raise e
        return

if __name__ == '__main__':
    new = NewGUI()
    # file = r'testdata\test.ARW'
    # process_file(
    #     file,
    #     raw_auto_bright_thr=0.001,
    #     statistics_performance=2,
    #     slopes=False,
    #     analyze_stars=True,
    #     plot_result=True,
    #     region_size=(1600,400),
    #     region=0,
    #     analyze_colors=1,
    #     check_input=False,
    #     return_colors_separately=True,
    # )



