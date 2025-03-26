import numpy as np


def calculate_velocity(
        coords: np.ndarray, t_sample: float = 1.0) -> np.ndarray:
    """
    Given an array of coordinates function calculates the velocities dividing
    the distance between two consecutive touch points over the duration between
    them (sampling rate average in miliseconds).

    Args:
        coords (np.ndarray): Array with coordinates.
        t_sample (float): Sampling frequency.

    Returns:
        np.ndarray: Array with velocities.
    """
    # Calculation of the first derivate
    d = np.diff(coords)
    d = np.insert(d, 0, 0) / t_sample
    return d


def calculate_speed(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Given two arrays of velocities this function calculates the combined speed.
    Formula: array(sqrt(dx^2 + dy^2)).

    Args:
        dx (np.ndarray): Array with X velocities.
        dy (np.ndarray): Array with Y velocities.

    Returns:
        np.ndarray: Array with combined speed.
    """

    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


class BaseGlobalFeaturesExtractor:
    """
    Class created for the calculation of the base feature set consisting of a
    maximum of 114 features. Of this 114 features: 97 features are from [1] and
    17 features are from [2].

    REFERENCES:

    [1] - Fierrez-Aguilar J., Nanni L., Lopez-Peñalba J., Ortega-Garcia J.,
    Maltoni D. (2005) An On-Line Signature Verification System Based on Fusion
    of Local and Global Information. In: Kanade T., Jain A., Ratha N.K. (eds)
    Audio- and Video-Based Biometric Person Authentication. AVBPA 2005. Lecture
    Notes in Computer Science, vol 3546. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/11527923_54

    [2] - R. Tolosana, R. Vera-Rodriguez, J. Fierrez and J. Ortega-Garcia,
    "Feature-based dynamic signature verification under forensic scenarios,"
    3rd International Workshop on Biometrics and Forensics (IWBF 2015), Gjovik,
    Norway, 2015, pp. 1-6, https://doi.org/10.1109/IWBF.2015.7110241.
    """

    def __init__(self, threshold_highr: int = 3, tsample: int = 20,
                 length_window: int = 30, threshold_v0: int = 1,
                 h_frame: int = 1920, v_frame: int = 1200):
        # Global variables
        self.THRESHOLD_HIGHR = threshold_highr  # high curvature points
        self.TSAMPLE = tsample  # Defaults 20 ms -> 50Hz sampling frequency
        self.LENGTH_WINDOW = length_window  # Minimum number of points
        self.THRESHOLD_V0 = threshold_v0  # xa detectar eventos v=0
        self.H_FRAME = h_frame  # horizontal space allowed for the test
        self.V_FRAME = v_frame  # vertical space allowed for the test

    def __weird_division(self, n, d):
        """
        Avoid a division by zero. If the denominator (d) is zero the function
        return 0.
        """
        return n / d if d else 0

    def __getAlfa(self, x1, y1, x2, y2):
        alfa = np.arctan2((y2-y1), (x2-x1))
        if alfa < 0:
            alfa += 2 * np.pi

        return alfa

    def __window(self, x: np.ndarray) -> np.ndarray:
        ANCHURA = 10

        maxs0 = np.array([0])
        for i in range(ANCHURA, len(x)-ANCHURA):
            x0 = x[(i-ANCHURA):(i+ANCHURA)]

            i_max_x0 = np.where(x0 == np.max(x0))[0] + (i-ANCHURA)
            if len(np.where(maxs0 == i_max_x0[0])[0]) == 0:
                maxs0 = np.concatenate((maxs0, [i_max_x0[0]]))

        maxs0 = maxs0[1:]

        maxs = np.array([0])
        tramo = np.array([maxs0[0]])
        i_off = 0
        for i in range(1, len(maxs0)):
            if maxs0[i] == maxs0[i-1]+1:
                tramo = np.concatenate((tramo, [maxs0[i]]))
            else:
                i_max_tramo = np.where(x[tramo] == np.max(x[tramo]))[0]
                maxs = np.concatenate((maxs, [maxs0[i_max_tramo[0] + i_off]]))

                tramo = np.array([maxs0[i]])
                i_off = i - 1

        i_max_tramo = np.where(x[tramo] == np.max(x[tramo]))[0]
        maxs = np.concatenate((maxs, [maxs0[i_max_tramo[0] + i_off]]))

        maxs = maxs[1:]
        vmaxs = np.sort(x[maxs])

        return vmaxs

    def __rms(self, x: np.ndarray) -> float:
        """
        RMS   Root-mean-square of a vector
        RMS(X) is the root-mean-square of X, namely:

        Xrms= (1/N* E(X(k)^2) {k=1...N})^1/2
        """
        return np.sqrt(np.power(x, 2).sum() / x.shape[0])

    def extract(
            self, x: np.ndarray, y: np.ndarray, timestamp: np.ndarray,
            time_features: bool = True, kinematic_features: bool = True,
            geometry_features: bool = True) -> np.ndarray:
        """
        This function calculates the 114 base features for a given
        sample. It receives the information in the form of Numpy arrays and
        returns an array with the calculated feature set. It allows to
        configure the type of features to be calculated.

        Args:
            x (np.ndarray): X-coordinates.
            y (np.ndarray): Y-coordinates.
            timestamp (np.ndarray): Time instants at which each sample was
                collected.
            time_features (bool, optional): Determines whether Time-like
                features should be calculated. Defaults to True.
            kinematic_features (bool, optional): Determines whether
                Kinematic-like features should be calculated. Defaults to True.
            geometry_features (bool, optional): Determines whether
                Geometry-like features should be calculated. Defaults to True.

        Raises:
            ValueError: Raise when the length of any input array is different
                from any other.

        Returns:
            np.ndarray: Base global features array.
        """

        if len(set(map(len,
                       [x, y, timestamp]))) != 1:
            raise ValueError(
                'All columns must have the same length and belong to the same'
                ' sample.')

        # Ignore warning when there is a division by zero
        np.seterr(divide='ignore', invalid='ignore')
        
        timestamp = np.subtract(timestamp, timestamp[0])

        # Numpy array to be returned
        vec_params = np.zeros(114)

        # Check if the child has drawn something
        if len(x) > 2:

            ###############################################################
            ##############        VARIABLES NEEDED        #################
            ###############################################################

            # x,y -> dx, dy (have one less sample)
            dx = x[1:] - x[:-1]
            dy = y[1:] - y[:-1]
            
            pte = dx / dy

            # x,y -> vx, vx, v
            vx = calculate_velocity(x, self.TSAMPLE)
            vy = calculate_velocity(y, self.TSAMPLE)
            v = calculate_speed(vx, vy)

            # vx, vy -> thetha
            theta = np.arctan2(vy, vx)

            # vx, vy, v -> ac, at, a
            ax = calculate_velocity(vx, self.TSAMPLE)
            ay = calculate_velocity(vy, self.TSAMPLE)
            at = calculate_velocity(v, self.TSAMPLE)
            thetad = calculate_velocity(theta, self.TSAMPLE)
            ac = v * thetad
            jx = calculate_velocity(ax, self.TSAMPLE)
            jy = calculate_velocity(ay, self.TSAMPLE)
            rcurva = v / thetad

            # ax, ay -> a
            a = np.abs(ax + 1j*ay)

            # jx, jy -> jerk
            jerk = np.abs(jx + 1j*jy)
            
            ts = timestamp[-1]      # total test duration (ms)
            tw = ts                 # total pendown duration 
            
            ###############################################################
            ##############        22 TIME FEATURES        #################
            ###############################################################

            if time_features:
                vec_params[0] = ts  # signature total duration ts
                vec_params[1] = np.nan
                vec_params[2] = np.nan
                vec_params[3] = np.count_nonzero(
                    vx > 0) * self.TSAMPLE / tw  # t(vx>0)/Tw
                vec_params[4] = np.count_nonzero(
                    vx < 0) * self.TSAMPLE / tw  # t(vx<0)/Tw
                vec_params[5] = np.count_nonzero(
                    vy > 0) * self.TSAMPLE / tw  # t(vy>0)/Tw
                vec_params[6] = np.count_nonzero(
                    vy < 0) * self.TSAMPLE / tw  # t(vy<0)/Tw
                vec_params[7:15] = np.nan
                vec_params[15] = (
                    len([item for item in rcurva if
                         item > self.THRESHOLD_HIGHR])
                    + len([item for item in rcurva if
                           item < (-self.THRESHOLD_HIGHR)])) \
                    * self.TSAMPLE / tw

                # Check that there are sufficient peaks in the X-coordinates to
                # differentiate between global and local maxima.
                if len(x) > self.LENGTH_WINDOW:
                    x_maxs = self.__window(x)

                    i_x_maxs_1 = np.where(x == x_maxs[0])[0]
                    # 1st max time
                    vec_params[16] = i_x_maxs_1[0] * self.TSAMPLE / tw

                    if len(x_maxs) > 1:
                        i_x_maxs_2 = np.where(x == x_maxs[1])[0]
                        # 2nd max time
                        vec_params[17] = i_x_maxs_2[0] * self.TSAMPLE / tw

                    if len(x_maxs) > 2:
                        i_x_maxs_3 = np.where(x == x_maxs[2])[0]
                        # 3rd max time
                        vec_params[18] = i_x_maxs_3[0] * self.TSAMPLE / tw

                else:
                    # only exists the global xmax
                    i_x_maxs_1 = np.where(x == np.max(x))[0]
                    vec_params[16] = i_x_maxs_1[0] * self.TSAMPLE / tw

                # Check that there are sufficient peaks in the Y-coordinates to
                # differentiate between global and local maxima.
                if len(y) > self.LENGTH_WINDOW:
                    y_maxs = self.__window(y)

                    i_y_maxs_1 = np.where(y == y_maxs[0])[0]
                    # 1st max time
                    vec_params[19] = i_y_maxs_1[0] * self.TSAMPLE / tw

                    if len(y_maxs) > 1:
                        i_y_maxs_2 = np.where(y == y_maxs[1])[0]
                        # 2nd max time
                        vec_params[20] = i_y_maxs_2[0] * self.TSAMPLE / tw

                    if len(y_maxs) > 2:
                        i_y_maxs_3 = np.where(y == y_maxs[2])[0]
                        # 3rd max time
                        vec_params[21] = i_y_maxs_3[0] * self.TSAMPLE / tw

                else:
                    # only  the global ymax
                    i_y_maxs_1 = np.where(y == np.max(y))[0]
                    vec_params[19] = i_y_maxs_1[0] * self.TSAMPLE / tw
                
            ###############################################################
            ##############      27 KINEMATIC FEATURES     #################
            ###############################################################

            if kinematic_features:
                vec_params[22] = np.mean(v) / np.max(v)  # v avg/vmax

                vec_params[23] = len(np.where(np.abs(vx) <= self.THRESHOLD_V0)[
                    0])  # total vx=0+ events recorded
                vec_params[24] = len(np.where(np.abs(vy) <= self.THRESHOLD_V0)[
                    0])  # total vy=0+ events recorded

                vec_params[25] = np.mean(v) / np.max(vx)  # v avg/vx,max
                vec_params[26] = np.mean(v) / np.max(vy)  # v avg/vy,max
                vec_params[27] = self.__rms(v) / np.max(v)  # norm rms v
                vec_params[28] = self.__rms(
                    ac) / np.max(a)  # norm rms centripetal a
                vec_params[29] = self.__rms(
                    at) / np.max(a)  # norm rms tangential a
                vec_params[30] = self.__rms(a) / np.max(a)  # norm rms a
                # norm integrated absolute ac
                vec_params[31] = np.sum(np.abs(ac)) / np.max(a)
                # norm xy speed correlation
                vec_params[32] = np.abs(np.sum(vx*vy)) / np.power(np.max(v), 2)

                vec_params[33] = np.std(vx)  # standard deviation of vx
                vec_params[34] = np.std(vy)  # standard deviation of vy
                vec_params[35] = np.std(ax)  # standard deviation of ax
                vec_params[36] = np.std(ay)  # standard deviation of ay

                vec_params[37] = np.mean(jerk)  # jerk,avg
                vec_params[38] = np.mean(jx)  # jerkx,avg
                vec_params[39] = np.mean(jy)  # jerky,avg
                vec_params[40] = np.max(jerk)  # jerk,max
                vec_params[41] = np.max(jx)  # jerkx,max
                vec_params[42] = np.max(jy)  # jerky,max
                vec_params[43] = self.__rms(jerk)  # jerk,rms

                i_jerk_max = np.where(jerk == np.max(jerk))[0]
                i_jx_max = np.where(jx == np.max(jx))[0]
                i_jy_max = np.where(jy == np.max(jy))[0]

                # norm instante de jerk,max
                vec_params[44] = i_jerk_max[0] * self.TSAMPLE / tw
                # norm instante de jx,max
                vec_params[45] = i_jx_max[0] * self.TSAMPLE / tw
                # norm instante de jy,max
                vec_params[46] = i_jy_max[0] * self.TSAMPLE / tw

                vec_params[112] = np.mean(v)  # average velocity
                vec_params[113] = np.mean(a)  # average aceleration

            ###############################################################
            ##############      32 GEOMETRY FEATURES      #################
            ###############################################################

            if geometry_features:
                # vec_params[65] = len(i_penups)-1  # n pen-ups
                vec_params[65] = np.nan
                
                d_qp, d_qn = 0, 0
                for i in range(len(dx)):
                    if np.sign(dx[i]/dy[i]) > 0:
                        d_qp += np.abs(dx[i] + 1j*dy[i])
                    if np.sign(dx[i]/dy[i]) < 0:
                        d_qn += np.abs(dx[i] + 1j*dy[i])

                    if i > 0:
                        if np.sign(dx[i]) != np.sign(dx[i-1]):
                            if np.sign(dx[i]) != 0:
                                # n of quadrant slope changes
                                vec_params[66] += 1
                        elif np.sign(dy[i]) != np.sign(dy[i-1]) \
                                and np.sign(dy[i]) != 0:
                            vec_params[66] += 1  # n of quadrant slope changes

                # writing distance in Q I,III / writing distance in Q II,IV
                vec_params[67] = d_qp / d_qn

                # t positive slopes / t negative slopes
                try:
                    vec_params[68] = np.count_nonzero(
                        pte > 0) / np.count_nonzero(pte < 0)
                except ZeroDivisionError:
                    vec_params[68] = 0.0
                
                vec_params[69] = np.nan
                
                # we look for all points of interest (at the extremes)
                amin = (np.max(y)-np.min(y)
                        ) * (np.max(x)-np.min(x))
                xmax = np.where(x == np.max(x))[0]
                xmin = np.where(x == np.min(x))[0]
                ymax = np.where(y == np.max(y))[0]
                ymin = np.where(y == np.min(y))[0]

                i_exts = np.concatenate((xmax, xmin, ymax, ymin))
                xsumando = x[i_exts]
                ysumando = y[i_exts]
                z_puntos = xsumando + 1j*ysumando

                # calculamos las distancias entre ellos (n^2 aunque no seran
                # muchos)
                d_z = np.zeros((len(z_puntos), len(z_puntos)))
                for i in range(len(z_puntos)):
                    for j in range(len(z_puntos)):
                        d_z[i][j] = np.abs(z_puntos[i]-z_puntos[j])

                # distancia max entre dos puntos, norm
                vec_params[70] = np.max(d_z) / amin
                
                vec_params[71:82] = np.nan
                
                xy_range = range(len(x)) 
                
                vec_params[82] = (
                    tw * np.mean(v)) / (np.max(x[xy_range]) -
                                        np.min(x[xy_range]))
                # lw length (recorrido)-to-height ratio
                vec_params[83] = (
                    tw * np.mean(v)) / (np.max(y[xy_range]) -
                                        np.min(y[xy_range]))
                vec_params[84] = (
                    np.max(x[xy_range]) - np.min(x[xy_range])) / self.H_FRAME
                vec_params[85] = (
                    np.max(y[xy_range]) - np.min(y[xy_range])) / self.V_FRAME
                # norm horizontal centroid xcn
                vec_params[86] = (np.mean(x[xy_range]) - np.min(x[xy_range])
                                  ) / np.mean(x[xy_range])

                xcg = np.min(x[xy_range]) + (np.max(x[xy_range]) -
                                             np.min(x[xy_range])) / 2
                ycg = np.min(y[xy_range]) + (np.max(y[xy_range]) -
                                             np.min(y[xy_range])) / 2
                x_cg = x[xy_range] - xcg
                y_cg = y[xy_range] - ycg

                for i in range(len(x_cg)):
                    if x_cg[i] > 0 and y_cg[i] > 0:
                        vec_params[87] += 1
                    if x_cg[i] < 0 and y_cg[i] > 0:
                        vec_params[88] += 1
                    if x_cg[i] < 0 and y_cg[i] < 0:
                        vec_params[89] += 1
                    if x_cg[i] > 0 and y_cg[i] < 0:
                        vec_params[90] += 1

                vec_params[87:91] /= len(x_cg)

                # Check that there are sufficient peaks in the X-coordinates to
                # differentiate between global and local maxima.
                try:
                    vec_params[91] = len(x_maxs)  # n local maximuxs for x
                    if len(x_maxs) > 1:
                        # norm 2o max
                        vec_params[92] = self.__weird_division(
                            (x_maxs[len(x_maxs) - 2] - x[i_pendowns[0]]),
                            shiftx)
                    if len(x_maxs) > 2:
                        # norm 3o max
                        vec_params[93] = self.__weird_division(
                            (x_maxs[len(x_maxs) - 3] - x[i_pendowns[0]]),
                            shiftx)
                except NameError:
                    vec_params[91] = 1.0

                # Check that there are sufficient peaks in the Y-coordinates to
                # differentiate between global and local maxima.
                try:
                    vec_params[94] = len(y_maxs)  # n local maximuxs for y
                    if len(y_maxs) > 1:
                        # norm 2o max
                        vec_params[95] = self.__weird_division(
                            (y_maxs[len(y_maxs) - 2] - y[i_pendowns[0]]),
                            shifty)

                    if len(y_maxs) > 2:
                        # norm 3o max
                        vec_params[96] = self.__weird_division((y_maxs[len(y_maxs) - 3] - y[i_pendowns[0]]), shifty)
                except NameError:
                    vec_params[94] = 1.0
                    
        # Set as np.nan those value that should not be calculated
        if not time_features:
            vec_params[0:22] = np.nan
        if not kinematic_features:
            vec_params[22:47] = np.nan
            vec_params[112:114] = np.nan
        # not direction_features:
        vec_params[47:65] = np.nan
        if not geometry_features:
            vec_params[65:97] = np.nan
        # not pressure_features
        vec_params[97:112] = np.nan

        # Delete np.nan values (types of features not calculated)
        #vec_params = vec_params[~np.isnan(vec_params)]

        return vec_params
