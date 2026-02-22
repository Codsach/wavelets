import numpy as np
import pywt
import cv2

def extract_dwt_features(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    features = []
    feature_names = []
    
    subbands = {
        "LL": LL,
        "LH": LH,
        "HL": HL,
        "HH": HH
    }
    
    for name, mat in subbands.items():
        features.extend([
            np.mean(mat),
            np.std(mat),
            np.var(mat)
        ])
        
        feature_names.extend([
            f"{name}_mean",
            f"{name}_std",
            f"{name}_var"
        ])
    
    return np.array(features), feature_names

def extract_wavelet_packet_features(image):
    wp = pywt.WaveletPacket2D(data=image, wavelet='haar', mode='symmetric', maxlevel=2)
    
    features = []
    feature_names = []
    
    nodes = wp.get_level(2, order='natural')
    
    for i, node in enumerate(nodes):
        mat = node.data
        
        features.extend([
            np.mean(mat),
            np.std(mat),
            np.var(mat)
        ])
        
        feature_names.extend([
            f"WP_node{i}_mean",
            f"WP_node{i}_std",
            f"WP_node{i}_var"
        ])
    
    return np.array(features), feature_names