
def get_attack_type(attack_type_id):
    """Convert numeric attack type ID to attack type name"""
    attack_type_mapping = {np.int64(0): 'BENIGN', np.int64(1): 'DoS GoldenEye', np.int64(2): 'DoS Hulk', np.int64(4): 'DoS slowloris', np.int64(3): 'DoS Slowhttptest'}
    return attack_type_mapping.get(attack_type_id, "Unknown")

def get_attack_type_id(attack_type_name):
    """Convert attack type name to numeric ID"""
    attack_type_mapping = {'BENIGN': np.int64(0), 'DDoS': np.int64(1), 'PortScan': np.int64(1), 'Bot': np.int64(1), 'Infiltration': np.int64(1), 'Web Attack � Brute Force': np.int64(1), 'Web Attack � XSS': np.int64(2), 'FTP-Patator': np.int64(1), 'SSH-Patator': np.int64(2), 'DoS slowloris': np.int64(4), 'DoS Slowhttptest': np.int64(3), 'DoS Hulk': np.int64(2), 'DoS GoldenEye': np.int64(1)}
    return attack_type_mapping.get(attack_type_name, -1)
