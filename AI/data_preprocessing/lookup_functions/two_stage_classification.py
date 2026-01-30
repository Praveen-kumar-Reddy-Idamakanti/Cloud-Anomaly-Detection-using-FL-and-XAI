
def get_attack_category(category_id):
    """Convert numeric category ID to attack category name"""
    category_mapping = {np.int64(0): 'Infiltration', np.int64(1): 'Other'}
    return category_mapping.get(category_id, "Unknown")

def get_attack_category_id(category_name):
    """Convert attack category name to numeric ID"""
    category_mapping = {'Normal': np.int64(0), 'DoS': np.int64(0), 'PortScan': np.int64(1), 'Botnet': np.int64(0), 'Infiltration': np.int64(0), 'Other': np.int64(1)}
    return category_mapping.get(category_name, -1)


def get_attack_type(attack_type_id):
    """Convert numeric attack type ID to attack type name"""
    attack_type_mapping = {np.int64(0): 'BENIGN', np.int64(1): 'DoS GoldenEye', np.int64(2): 'DoS Hulk', np.int64(4): 'DoS slowloris', np.int64(3): 'DoS Slowhttptest'}
    return attack_type_mapping.get(attack_type_id, "Unknown")

def get_attack_type_id(attack_type_name):
    """Convert attack type name to numeric ID"""
    attack_type_mapping = {'BENIGN': np.int64(0), 'DDoS': np.int64(1), 'PortScan': np.int64(1), 'Bot': np.int64(1), 'Infiltration': np.int64(1), 'Web Attack � Brute Force': np.int64(1), 'Web Attack � XSS': np.int64(2), 'FTP-Patator': np.int64(1), 'SSH-Patator': np.int64(2), 'DoS slowloris': np.int64(4), 'DoS Slowhttptest': np.int64(3), 'DoS Hulk': np.int64(2), 'DoS GoldenEye': np.int64(1)}
    return attack_type_mapping.get(attack_type_name, -1)


def two_stage_classification(binary_prediction, category_prediction=None, attack_type_prediction=None):
    """
    Two-stage classification system for anomaly detection
    
    Args:
        binary_prediction: 0 (Normal) or 1 (Anomaly)
        category_prediction: Attack category ID (optional)
        attack_type_prediction: Specific attack type ID (optional)
    
    Returns:
        dict: Classification results at different levels
    """
    result = {
        'is_anomaly': bool(binary_prediction),
        'classification_level': 'Normal' if binary_prediction == 0 else 'Anomaly'
    }
    
    if binary_prediction == 1:  # If anomaly detected
        if category_prediction is not None:
            result['attack_category'] = get_attack_category(category_prediction)
            result['classification_level'] = result['attack_category']
            
            if attack_type_prediction is not None:
                result['attack_type'] = get_attack_type(attack_type_prediction)
                result['classification_level'] = result['attack_type']
    
    return result

def interpret_anomaly_result(binary_pred, category_pred=None, type_pred=None):
    """
    Interpret anomaly detection results with confidence levels
    """
    if binary_pred == 0:
        return {
            'status': 'Normal',
            'confidence': 'High',
            'details': 'No malicious activity detected'
        }
    
    result = {
        'status': 'Anomaly Detected',
        'confidence': 'High'
    }
    
    if category_pred is not None:
        category = get_attack_category(category_pred)
        result['attack_category'] = category
        result['details'] = f'Malicious activity detected: {category}'
        
        if type_pred is not None:
            attack_type = get_attack_type(type_pred)
            result['attack_type'] = attack_type
            result['details'] = f'Specific attack detected: {attack_type} ({category})'
    else:
        result['details'] = 'Malicious activity detected, but specific type unknown'
    
    return result
