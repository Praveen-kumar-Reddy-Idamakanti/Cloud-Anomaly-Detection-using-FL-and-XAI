
def get_attack_category(category_id):
    """Convert numeric category ID to attack category name"""
    category_mapping = {np.int64(0): 'Infiltration', np.int64(1): 'Other'}
    return category_mapping.get(category_id, "Unknown")

def get_attack_category_id(category_name):
    """Convert attack category name to numeric ID"""
    category_mapping = {'Normal': np.int64(0), 'DoS': np.int64(0), 'PortScan': np.int64(1), 'Botnet': np.int64(0), 'Infiltration': np.int64(0), 'Other': np.int64(1)}
    return category_mapping.get(category_name, -1)
