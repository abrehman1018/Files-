import numpy as np

# Updated threat types
threat_types = [
    'x-mitre-matrix', 'course-of-action', 'malware', 'tool', 'x-mitre-tactic',
    'attack-pattern', 'x-mitre-data-component', 'campaign', 'intrusion-set', 'x-mitre-data-source'
]

# Map each threat type to a unique numerical value
label_map = {
    'x-mitre-matrix': 0, 
    'course-of-action': 1, 
    'malware': 2, 
    'tool': 3, 
    'x-mitre-tactic': 4,
    'attack-pattern': 5, 
    'x-mitre-data-component': 6, 
    'campaign': 7, 
    'intrusion-set': 8, 
    'x-mitre-data-source': 9
}

# Convert labels to numerical format
labels = np.array([label_map[t] for t in threat_types])

# Save labels to a file
np.save('labels.npy', labels)

print("Threat types have been mapped and saved to 'labels.npy'.")
