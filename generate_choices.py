from random import choice, sample
def generate_sentences_with_details_corrected(num_examples=50):
    examples = []

    # Adjusted examples with keys and values switched
    concrete_examples = {
        "one person running": "Running or jumping",
        "one person riding a bicycle": "Using any non-walking movement such as riding a bicycle or scooter, skateboarding",
        "someone sprinting": "Moving with unusual speed such as very fast or in hurry",
        "a person lying on the ground": "Person lying or bending down on the ground",
        "two people fighting": "Aggressive or unusual movements such as fighting or pushing",
        "a person loitering around a trash bin": "Loitering around trash bin",
        "someone climbing a tree": "Climbing on trees or into manhole/utility access points",
        "a person tampering with a manhole cover": "Tampering with manhole covers",
        "someone loitering suspiciously near a trash bin": "Loitering suspiciously around the trash bin",
        "individuals engaging in a physical altercation": "Engaging in physical altercation",
        "a person lingering near a sensitive area": "Lingering without a clear purpose near a sensitive area",
        "someone falling down unexpectedly": "Falling unexpectedly",
        "an ajar manhole cover": "Missing, removed, or ajar manhole covers",
        "a vehicle parked": "Ground transportations such as vehicles, vans, bicycles",
        "an unattended bag": "Unattended bags or objects (security threat)",
        "a large unattended luggage": "Unusual objects (vehicles, large unattended luggage, etc.)",
        "an unauthorized sign": "Unauthorized posters or signs",
        "a damaged manhole cover": "Damaged manhole cover or trees",
        "abandoned items in a park": "Abandoned items in a public area",
        "a bright yellow object": "Yellow objects (when not commonly seen)",
        "graffiti on a building window": "Graffiti on building windows or vandalism to a fence",
        "obstructions on a pedestrian path": "Obstructions on pedestrian crossing",
        "a fire hydrant spraying water": "Fire hydrant spraying water without presence of emergency services",
        "an unidentified object near a building": "Unidentified objects near the tall structure",
        "smoke coming from a manhole": "Smoke or fire coming from a utility access point",
        "objects falling from a building": "Objects thrown or falling from a height"
    }

    # Determine if each concrete example is a human activity or an environmental object
    human_activities_keys = list(concrete_examples.keys())[:12]  # First 12 are human activities
    environmental_objects_keys = list(concrete_examples.keys())[12:]  # Remaining are environmental objects

    for _ in range(num_examples):
        # Randomly place the normal sentence among A, B, C, D
        normal_position = choice(['A', 'B', 'C', 'D'])
        positions = ['A', 'B', 'C', 'D']
        sentences = {pos: "" for pos in positions}

        # Assign the normal sentence to its position
        sentences[
            normal_position] = "D. Normal, since no rules for anomaly human activities or environmental objects match."

        # Prepare combined list of anomalies with concrete examples
        combined_anomalies = list(concrete_examples.items())

        # Fill the other positions with anomalies
        selected_anomalies = sample(combined_anomalies, 3)  # Ensure unique anomalies are selected
        for pos in positions:
            if sentences[pos] == "":  # If this position is not the normal sentence
                concrete_example, rule = selected_anomalies.pop()
                activity_or_object = "anomaly human activities" if concrete_example in human_activities_keys else "anomaly environmental objects"
                sentences[pos] = f"Anomaly, since '{concrete_example}' matches {activity_or_object} '{rule}'."

        # Re-assign the correct sentence for normal position based on the random selection
        for pos in positions:
            if pos == normal_position:
                sentences[pos] = "Normal, since no rules for anomaly human activities or environmental objects match."

        # Construct the final sentence
        ordered_sentences = [f"{pos}. {sentences[pos]}" for pos in positions if sentences[pos]]
        examples.append(" ".join(ordered_sentences))

    return examples


# Generate corrected examples with the adjusted order of keys and values
generated_sentences_50_corrected = generate_sentences_with_details_corrected(50)
file_path_50_corrected = 'SHTech/anomaly_and_normal_activities_50_examples_corrected.txt'
with open(file_path_50_corrected, 'w') as file:
    for sentence in generated_sentences_50_corrected:
        file.write(sentence + "\n")
