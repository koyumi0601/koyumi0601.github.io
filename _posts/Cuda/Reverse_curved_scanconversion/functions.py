import json, re, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator

max8bitsInt = 2**8 - 1
max16bitsInt = 2**16 - 1

def rgb_to_gray_lightness(imgRgb): # Lightness method
    return (np.min(imgRgb[..., :3], axis=-1) + np.max(imgRgb[..., :3], axis=-1)) * 0.5

def rgb_to_gray_average(imgRgb): # Average method
    return np.dot(imgRgb[..., :3], [0.3333, 0.3333, 0.3333])

def rgb_to_gray_luminosity(imgRgb): # Luminosity method(general method), same result with tensorflow.image.rgb_to_grayscale()
    return np.dot(imgRgb[..., :3], [0.2989, 0.5870, 0.1140])

def load_image_as_rgb_and_gray8bits(fileName, method="luminosity"):
    try:
        imgRgb = plt.imread(fileName)
    except Exception:
        raise Exception("Unsupported image format of the file '{}'.".format(fileName))
    if len(imgRgb.shape) == 2:
        return imgRgb, imgRgb
    else:
        if imgRgb.dtype == np.uint8:
            imgRgb = imgRgb.astype(np.float32) / max8bitsInt
        elif imgRgb.dtype == np.uint16:
            imgRgb = imgRgb.astype(np.float32) / max16bitsInt
        elif imgRgb.dtype == np.float16:
            imgRgb = imgRgb.astype(np.float32)
        elif imgRgb.dtype == np.float32 or imgRgb.dtype == np.float64:
            pass
        else:
            raise ValueError("Unsupported image data type")
        if method == "lightness":
            return imgRgb, np.around(rgb_to_gray_lightness(imgRgb) * max8bitsInt).astype("uint8")
        elif method == "average":
            return imgRgb, np.around(rgb_to_gray_average(imgRgb) * max8bitsInt).astype("uint8")
        elif method == "luminosity":
            return imgRgb, np.around(rgb_to_gray_luminosity(imgRgb) * max8bitsInt).astype("uint8")
        else:
            raise ValueError("Unsupported rgb2gray method")

def select_dir_or_file(statement = "Select Something", target="dir", select_filter="All Files (*)", isMuliple=False, isFullPath=False):
    """
    파일 또는 디렉터리를 선택하는 대화 상자를 열고 선택된 항목의 경로를 반환합니다.

    매개변수:
    statement (str): 대화 상자에 표시할 메시지입니다. 기본값은 "Select Something"입니다.
    target (str): 선택할 대상 유형입니다. "dir" (디렉터리) 또는 "file" (파일) 중 하나여야 합니다. 기본값은 "dir"입니다.
    select_filter (str): 파일 대화 상자의 파일 필터를 지정하는 문자열입니다. 기본값은 "All Files (*)"입니다.
    isMuliple (bool): 여러 파일을 선택할 수 있는지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
    isFullPath (bool): 선택한 항목의 전체 경로를 반환할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.

    반환 값:
    선택한 항목(파일 또는 디렉터리)의 경로를 나타내는 문자열 또는 문자열 목록을 반환합니다. isMuliple이 True이면 선택한 모든 항목의 경로가 포함된 목록을 반환합니다.

    사용 예제:
    # 단일 디렉터리 선택
    selected_dir = select_dir_or_file(target="dir")

    # 단일 파일 선택
    selected_file = select_dir_or_file(target="file")

    # 여러 파일 선택 (전체 경로 포함)
    selected_files = select_dir_or_file(target="file", isMuliple=True, isFullPath=True)

    # 파일 필터 지정
    selected_filtered_file = select_dir_or_file(target="file", select_filter="Text Files (*.txt)")

    # 선택한 항목의 경로 출력
    print("Selected Item:", selected_dir)
    """
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWidgets import QFileDialog
    app = QApplication([])
    if target.lower() == "dir":
        path = QFileDialog.getExistingDirectory(None, statement, os.getcwd())
    elif target.lower() == "file":
        if isMuliple:
            if isFullPath:
                path = os.path.abspath(QFileDialog.getOpenFileNames(None, statement, os.getcwd(), filter=select_filter)[0]) # get only filename list
            else:
                path = QFileDialog.getOpenFileNames(None, statement, os.getcwd(), filter=select_filter)[0] # get only filename list
        else:
            if isFullPath:
                path = os.path.abspath(QFileDialog.getOpenFileName(None, statement, os.getcwd(), filter=select_filter)[0]) # get only filename list
            else:
                path = QFileDialog.getOpenFileName(None, statement, os.getcwd(), filter=select_filter)[0] # get only filename list
    app.quit()
    return path

def list_files_with_wildcard(path, wildcard):
    all_files = os.listdir(path)
    pattern = re.compile(wildcard.replace(".", "\.").replace("*", ".*"))
    return [filename for filename in all_files if pattern.match(filename)]

def flatten_list(m):
    return re.sub(',', ', ', re.sub('(?s)\s+', '', m.group(0)))

def write_json(aDict, fileName=None):
    aStr = json.dumps(aDict, indent=4)
    aStr = re.sub('(?s)\[([^{^}^\[^\]^;]+)\]', flatten_list, aStr)
    aStr = aStr.replace('[', '[ ').replace(']', ' ]')
    if fileName!=None:
        with open(fileName, 'w', encoding="UTF8") as fileToWrite:
            fileToWrite.write(aStr)
            fileToWrite.close()

def read_json(fileName):
    with open(fileName, "r") as fileToRead:
        aData = json.load(fileToRead, strict = False)
        fileToRead.close()
    return aData

def load_slices(pathname):
    from collections import OrderedDict
    slicesFileNameList = list_files_with_wildcard(pathname, "slice_*.png")
    sliceInfoFileName = list_files_with_wildcard(pathname, "slices_info_*.json")[0]
    slices = OrderedDict()
    for sliceFileName in slicesFileNameList:
        _, tmpImageGray = load_image_as_rgb_and_gray8bits(os.path.join(pathname, sliceFileName), method="luminosity")
        slices[os.path.splitext(sliceFileName)[0]] = tmpImageGray.T
    sliceInfo = read_json(os.path.join(pathname, sliceInfoFileName))
    return [slices, sliceInfo]

def draw3DVolume(volume, label="xyz", title="", threshold=127):
    volumeTF = np.array(volume).astype("bool")
    volumeTF[np.isnan(volume)] = False
    volumeTF[np.array(volume) < threshold] = False
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(volumeTF, facecolors="red")
    ax.set_xlabel(f"{label[0]} axis")
    ax.set_ylabel(f"{label[1]} axis")
    ax.set_zlabel(f"{label[2]} axis")
    ax.set_title(title)
    ax.set_box_aspect((volume.shape[0], volume.shape[1], volume.shape[2]))

def checkSymmetry(volume, maxVal):
    isSymmetry = True
    problemList = list()
    profileDict = {"x" : volume[:, int(volume.shape[1]/2), int(volume.shape[2]/2)],
                   "y" : volume[int(volume.shape[0]/2), :, int(volume.shape[2]/2)],
                   "z" : volume[int(volume.shape[0]/2), int(volume.shape[1]/2), :]}
    for key, profile in profileDict.items():
        numFrontZeros = np.argwhere(profile != 0)[0][0]
        numRearZeros = len(profile)-(np.argwhere(profile == maxVal)[-1][0]+1)
        if not(numFrontZeros == numRearZeros):
            isSymmetry = False
            problemList.append(f"{key}: numFrontZeros = {numFrontZeros}, numRearZeros = {numRearZeros}")
    return isSymmetry, problemList

def updateVolumeAsSymmetry(volume):
    x, y, z= volume.shape
    symmetricVolume = np.zeros(volume.shape)
    if np.mod(x, 2):
        symmetricVolume[:x//2+1, :, :] = volume[:x//2+1, :, :]
        symmetricVolume[x//2:, :, :] = np.flip(volume[:x//2+1, :, :], axis=0)
        symmetricVolume[x//2, :, :] = np.mean(volume[x//2-1:x//2+2, :, :], axis=0)
    else:
        symmetricVolume[:x//2, :, :] = volume[:x//2, :, :]
        symmetricVolume[x//2:, :, :] = np.flip(volume[:x//2, :, :], axis=0)
    if np.mod(y, 2):
        symmetricVolume[:, :y//2+1, :] = symmetricVolume[:, :y//2+1, :]
        symmetricVolume[:, y//2:, :] = np.flip(symmetricVolume[:, :y//2+1, :], axis=1)
        symmetricVolume[:, y//2, :] = np.mean(volume[:, y//2-1:y//2+2, :], axis=1)
    else:
        symmetricVolume[:, :y//2, :] = symmetricVolume[:, :y//2, :]
        symmetricVolume[:, y//2:, :] = np.flip(symmetricVolume[:, :y//2, :], axis=1)
    if np.mod(z, 2):
        symmetricVolume[:, :, :z//2+1] = symmetricVolume[:, :, :z//2+1]
        symmetricVolume[:, :, z//2:] = np.flip(symmetricVolume[:, :, :z//2], axis=2)
        symmetricVolume[:, :, z//2] = np.mean(volume[:, :, z//2-1:z//2+2], axis=2)
    else:
        symmetricVolume[:, :, :z//2] = symmetricVolume[:, :, :z//2]
        symmetricVolume[:, :, z//2:] = np.flip(symmetricVolume[:, :, :z//2], axis=2)
    return symmetricVolume

def getVolume(gridSize, type="cube", dtype="bool"):
    x, y, z = np.indices(gridSize)
    if type == "cube":
        volume = (((x < gridSize[0]*0.75) & (x >= gridSize[0]*0.25)) & ((y < gridSize[1]*0.75) & (y >= gridSize[1]*0.25)) & ((z < gridSize[2]*0.90) & (z >= gridSize[2]*0.10)))
    elif type == "sphere":
        volume = (np.sqrt((x - gridSize[0]/2)**2 + (y - gridSize[1]/2)**2 + (z - gridSize[2]/2)**2) <= np.min(gridSize)*0.35)
    elif type == "ellipsoid":
        volume = (np.sqrt(((x - gridSize[0]/2)/(gridSize[0]*0.275))**2 + ((y - gridSize[1]/2)/(gridSize[1]*0.250))**2 + ((z - gridSize[2]/2)/(gridSize[2]*0.375))**2) < 1.0)
    elif type == "cylinder":
        volume = ((np.sqrt((y - gridSize[1]/2)**2 + (z - gridSize[2]/2)**2) < np.min(gridSize)*0.35) & ((x < gridSize[0]*0.80) & (x > gridSize[0]*0.20)))
    return volume.astype(dtype)


def slice_datacube(cube, lotCenter, rZX, slcPlane, outPlaneSize, oversampled=1, fill=np.nan, isInterp=True, isIdxNormalized=False, isExportOut=False, dirname="", outType=float, scaler=1.0):
    """Get a 2D slices from a 3-D array.
    Parameters:
        - cube: 3D array, assumed shape (nx, ny, nz).
        - lotCenter: shape (3,) with coordinates of lotCenter. can be float. 
        - rZX: Rotated angle of Z, and X axis 
            so, eXY= unit vectors, shape (2, 3) - for X and Y axes of the slice.
            (unit vectors must be orthogonal; normalization is optional).
        - outPlaneSize: size tuple of output array (mU or mV or mW, mU or mV or mW except of previous one) - int.
        - slcPlane: target slicing plane on (u,v,w) coordinate
        - fill: value to use for out-of-range points.
        - interp: whether to interpolate (rather than using 'nearest')
    Return:
        - slice: array, shape (outPlaneSize).
    """
    unitX = 1.0 / (cube.shape[0] - 1)
    unitY = 1.0 / (cube.shape[1] - 1)
    unitZ = 1.0 / (cube.shape[2] - 1)
    lotCenter = np.array(lotCenter, dtype=float)
    assert lotCenter.shape == (3,)
    Rz = rZX[0]
    Rx = rZX[1]
    cz, sz = np.cos(Rz), np.sin(Rz)
    cx, sx = np.cos(Rx), np.sin(Rx)
    Rmz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) # [u=[x,y,z],v=[x,y,z],w=[x,y,z]] for z axis rotate
    Rmx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) # [u=[x,y,z],v=[x,y,z],w=[x,y,z]] for x axis rotate
    eXY = (Rmx @ Rmz).T[:2]
    eXY = np.array(eXY) / np.linalg.norm(eXY, axis=1)[:, np.newaxis]
    if not np.isclose(eXY[0] @ eXY[1], 0, atol=1e-6):
        raise ValueError(f"eX and eY not orthogonal.")
    # R: rotation matrix: data_coords = lotCenter + R @ slice_coords
    eZ = np.cross(eXY[0], eXY[1])
    R = np.array([eXY[0], eXY[1], eZ], dtype=np.float32).T # rotation matrix
    N = R.T # normal vector matrix of plane
    # setup slice points P with coordinates
    mX, mY = int(outPlaneSize[0]), int(outPlaneSize[1])
    Xs = np.arange(0.5-mX * oversampled/2, 0.5+mX * oversampled/2)
    Ys = np.arange(0.5-mY * oversampled/2, 0.5+mY * oversampled/2)
    assert type(slcPlane) == dict
    slicePlaneName = list(slcPlane.keys())[0].lower() # 2 permutations of u,v,w ex) uv, vu, vw, wv, uw, wu
    planeLocs = list(slcPlane.values())[0]
    assert slicePlaneName in ["uv", "vu", "vw", "wv", "uw", "wu"]
    assert planeLocs
    slicePlaneName = slicePlaneName.replace("u", "0").replace("v", "1").replace("w", "2")
    slices = list() # for verification
    planeOrigin = list() # for verification
    planeVector = list() # for verification
    aDict = dict()
    aDict_quat = dict()
    aTmpDict = dict()
    aTmpDict_quat = dict()
    for k, loc in enumerate(planeLocs):
        print (k)
        PP = np.ones((3, mX * oversampled, mY * oversampled), dtype=np.float32) * loc
        PP[int(slicePlaneName[0]), :, :] = Xs.reshape(mX * oversampled, 1)
        PP[int(slicePlaneName[1]), :, :] = Ys.reshape(1, mY * oversampled)
        if not("u" in slicePlaneName):
            planeNormVector = N[0]
            planeNormOrigin = (R.dot(np.array([loc, 0.0, 0.0])) + np.array(lotCenter)) * scaler
            osFactor = [1.0, oversampled, oversampled]
        elif not("v" in slicePlaneName):
            planeNormVector = N[1]
            planeNormOrigin = (R.dot(np.array([0.0, loc, 0.0])) + np.array(lotCenter)) * scaler
            osFactor = [oversampled, 1.0, oversampled]
        elif not("w" in slicePlaneName):
            planeNormVector = N[2]
            planeNormOrigin = (R.dot(np.array([0.0, 0.0, loc])) + np.array(lotCenter)) * scaler
            osFactor = [oversampled, oversampled, 1.0]
        if isIdxNormalized:
            planeNormOrigin = planeNormOrigin * np.array([unitX, unitY, unitZ])
        else:
            unitX, unitY, unitZ = 1.0, 1.0, 1.0
        # Transform to data coordinates (x, y, z) - idx.shape == (3, mX, mY)
        if isInterp:
            idx = (np.einsum('il,ljk->ijk', R, PP / np.array(osFactor).reshape(3, 1, 1))  + lotCenter.reshape(3, 1, 1))
            slice = map_coordinates(cube, idx, order=1, mode='constant', cval=fill)
        else:
            idx = np.einsum('il,ljk->ijk', R, PP / np.array(osFactor).reshape(3, 1, 1))  + (0.5 + lotCenter.reshape(3, 1, 1))
            idx = idx.astype(np.int16)
            # Find out which coordinates are out of range - shape (mX, mY)
            badpoints = np.any([
                idx[0, :, :] < 0,
                idx[0, :, :] >= cube.shape[0], 
                idx[1, :, :] < 0,
                idx[1, :, :] >= cube.shape[1], 
                idx[2, :, :] < 0,
                idx[2, :, :] >= cube.shape[2], 
                ], axis=0)
            idx[:, badpoints] = 0
            slice = cube[idx[0], idx[1], idx[2]]
            slice[badpoints] = fill
        if outType == int:
            slice = np.round(slice).astype("int")
        slices.append(slice)
        planeOrigin.append(planeNormOrigin.tolist())
        planeVector.append(planeNormVector.tolist())
        
        # because it is normal vector of a plane, which is rotated by this rotations
        roll = Rx - np.pi
        pitch = 0.0 - np.pi
        yaw = Rz - np.pi
        
        temp_quaternion_rotation = np.array([np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2),
                                             np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2),
                                             np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2),
                                             np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)])

        aTmpDict["slice_%04d" % k] = planeNormOrigin.tolist() + planeNormVector.tolist()
        aTmpDict_quat["slice_%04d" % k] = [k] + planeNormOrigin.tolist() + temp_quaternion_rotation.tolist()

        # if np.sum(np.abs(N -np.round(quaternion_rotation_matrix(temp_quaternion_rotation), 8))) > 0.000001:
        #     print(np.round(quaternion_rotation_matrix(temp_quaternion_rotation), 8))

        if isExportOut:
            if not(os.path.isdir(dirname)):
                os.mkdir(dirname)
            Image.imsave(os.path.join(".", dirname, "slice_%04d" % k + ".png"), slice.T, cmap="gray")
    aDict["unitFactor"] = [unitX, unitY, unitZ]
    aDict["planeOverSampleRate"] = oversampled
    aDict["slices"] = aTmpDict
    aDict_quat["unitFactor"] = [unitX, unitY, unitZ]
    aDict_quat["planeOverSampleRate"] = oversampled
    aDict_quat["slices"] = aTmpDict_quat

    if isExportOut:
        write_json(aDict, fileName=os.path.join(".", dirname, f"slices_info_{len(planeLocs)}ea.json"))
        write_json(aDict_quat, fileName=os.path.join(".", dirname, f"slices_info_{len(planeLocs)}ea_quaternion.json"))
        write_json(cube.tolist(), fileName=os.path.join(".", dirname, f"GroundTruth.json"))
    return slices, planeOrigin, planeVector

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix


def recon_volume(slices, sliceInfo, outType=float, istotalInterp=False, isDebug=False):
    import time
    # get info
    keyList = list(slices.keys())
    unitLength = sliceInfo["unitLength"]
    planeOverSampleRate = sliceInfo["planeOverSampleRate"]
    numSlices = len(slices)
    planeSize = list(slices.values())[0].shape
    # init v, w
    v = np.linspace(-(planeSize[0]-1)/2, (planeSize[0]-1)/2, planeSize[0], dtype=float) / ((planeSize[0]-1)/2) * sliceInfo["slices"][keyList[0]][2] / unitLength[0]
    w = np.linspace(-(planeSize[1]-1)/2, (planeSize[1]-1)/2, planeSize[1], dtype=float) / ((planeSize[0]-1)/2) * sliceInfo["slices"][keyList[0]][2] / unitLength[0]
    if isDebug:
        plt.figure()
        ax = plt.axes(projection='3d')
    # rotate
    X = np.linspace(0, max(planeSize)/planeOverSampleRate-1, int(max(planeSize)/planeOverSampleRate), dtype=int)
    Y = np.linspace(0, max(planeSize)/planeOverSampleRate-1, int(max(planeSize)/planeOverSampleRate), dtype=int)
    Z = np.linspace(0, max(planeSize)/planeOverSampleRate-1, int(max(planeSize)/planeOverSampleRate), dtype=int)
    accumVolume = np.zeros((len(X), len(Y), len(Z)))
    print(X)
    Xi, Yi, Zi = np.meshgrid(X, Y, Z, indexing='ij')

    if not(istotalInterp):
        for n in range(numSlices-1):
            # slice k
            slice0 = slices[keyList[n]]
            tmpSliceInfo0 = sliceInfo["slices"][keyList[n]]
            tmp_nxnynz_u_0 = tmpSliceInfo0[3:]
            tmp_nxnynz_v_0 = [tmp_nxnynz_u_0[1]*-1, tmp_nxnynz_u_0[0], tmp_nxnynz_u_0[2]]
            eUV0 = np.array([tmp_nxnynz_u_0, tmp_nxnynz_v_0])
            eUV0 = np.array(eUV0)/np.linalg.norm(eUV0, axis=1)[:, np.newaxis]
            tmp_R0 = np.array([eUV0[0], eUV0[1], np.cross(eUV0[0], eUV0[1]).tolist()]).T
            tmp_C0 = tmpSliceInfo0[:3] / np.array(unitLength)
            # slice k+1
            slice1 = slices[keyList[n+1]]
            tmpSliceInfo1 = sliceInfo["slices"][keyList[n+1]]
            tmp_nxnynz_u_1 = tmpSliceInfo1[3:]
            tmp_nxnynz_v_1 = [tmp_nxnynz_u_1[1]*-1, tmp_nxnynz_u_1[0], tmp_nxnynz_u_1[2]]
            eUV1 = np.array([tmp_nxnynz_u_1, tmp_nxnynz_v_1])
            eUV1 = np.array(eUV1)/np.linalg.norm(eUV1, axis=1)[:, np.newaxis]
            tmp_R1 = np.array([eUV1[0], eUV1[1], np.cross(eUV1[0], eUV1[1]).tolist()]).T
            tmp_C1 = tmpSliceInfo1[:3] / np.array(unitLength)
            x = list()
            y = list()
            z = list()
            val = list()
            st = time.time()
            for vi in v:
                for wi in w:
                    tmp_xyz0 = tmp_R0 @ np.array([0, vi, wi]) + tmp_C0
                    x.append(tmp_xyz0[0])
                    y.append(tmp_xyz0[1])
                    z.append(tmp_xyz0[2])
                    val.append(slice0[v == vi, w == wi])

                    tmp_xyz1 = tmp_R1 @ np.array([0, vi, wi]) + tmp_C1
                    x.append(tmp_xyz1[0])
                    y.append(tmp_xyz1[1])
                    z.append(tmp_xyz1[2])
                    val.append(slice1[v == vi, w == wi])
                    if isDebug:
                        if slice[v == vi, w == wi] > 0:
                            ax.scatter(tmp_xyz0[0], tmp_xyz0[1], tmp_xyz0[2])
                            ax.scatter(tmp_xyz1[0], tmp_xyz1[1], tmp_xyz1[2])
            print(f"    collecting time {np.round(time.time() - st)}")
            # interpolate
            # st = time.time()
            tri = Delaunay(np.column_stack([x, y, z]))
            # tri = np.column_stack([x, y, z])
            # print(f"    Delaunay time {np.round(time.time() - st)}")
            # st = time.time()
            interp_func = LinearNDInterpolator(tri, val)
            print(f"    LinearNDInterpolator time {np.round(time.time() - st, 2)}")
            print(f"    Reconstructing volume is progressed at {round(n/(numSlices-1-1)*100, 2)} % ...")
            volume = np.squeeze(interp_func(Xi, Yi, Zi))
            volume[np.isnan(volume)] = 0
            accumVolume = accumVolume + volume
    else:
        x = list()
        y = list()
        z = list()
        val = list()
        for n in range(numSlices):
            # slice k
            slice0 = slices[keyList[n]]
            tmpSliceInfo0 = sliceInfo["slices"][keyList[n]]
            tmp_nxnynz_u_0 = tmpSliceInfo0[3:]
            tmp_nxnynz_v_0 = [tmp_nxnynz_u_0[1]*-1, tmp_nxnynz_u_0[0], tmp_nxnynz_u_0[2]]
            eUV0 = np.array([tmp_nxnynz_u_0, tmp_nxnynz_v_0])
            eUV0 = np.array(eUV0)/np.linalg.norm(eUV0, axis=1)[:, np.newaxis]
            tmp_R0 = np.array([eUV0[0], eUV0[1], np.cross(eUV0[0], eUV0[1]).tolist()]).T
            tmp_C0 = tmpSliceInfo0[:3] / np.array(unitLength)
            for vi in v:
                for wi in w:
                    tmp_xyz0 = tmp_R0 @ np.array([0, vi, wi]) + tmp_C0
                    x.append(tmp_xyz0[0])
                    y.append(tmp_xyz0[1])
                    z.append(tmp_xyz0[2])
                    val.append(slice0[v == vi, w == wi])
                    if isDebug:
                        if slice[v == vi, w == wi] > 0:
                            ax.scatter(tmp_xyz0[0], tmp_xyz0[1], tmp_xyz0[2])
        # interpolate
        # tri = Delaunay(np.column_stack([x, y, z]))
        # interp_func = LinearNDInterpolator(tri, val)
        interp_func = LinearNDInterpolator(np.column_stack([x, y, z]), val)
        print(f"    Reconstructing volume is started ...")
        volume = np.squeeze(interp_func(Xi, Yi, Zi))
        volume[np.isnan(volume)] = 0
        accumVolume = volume
    accumVolume = np.clip(accumVolume, 0, 255)
    print(f"    Reconstructing volume is done.")


    if isDebug:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(0, max(planeSize)-1)
        ax.set_ylim(0, max(planeSize)-1)
        ax.set_zlim(0, max(planeSize)-1)
        ax.set_box_aspect((1, 1, 1))
        plt.show()
    if outType == int:
        return np.round(accumVolume).astype("int")
    else:
        return accumVolume
