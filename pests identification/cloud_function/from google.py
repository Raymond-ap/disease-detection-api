from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

model = None
interpreter = None
input_index = None
output_index = None

CLASS_NAMES = ["ants", "bees", "beetle", "caterpillar", "earthworms", "earwig", "grasshopper",
              "moth", "slug", "snail", "wasp", "weevil"]

CLOUD_BUCKET_NAME = 'cs-tf-modal'

# Definately not the best way to do this, but it works
PEST_INFO = [
    {
        "name": "ants",
        "description": "Ants are social insects that belong to the family Formicidae. They are highly organized and live in colonies with distinct roles, including workers, soldiers, and a queen. Ants are attracted to food sources and often forage in search of sugary or protein-rich substances.",
        "management": [
            "Sanitation: Keep your home and surroundings clean to remove potential food sources.",
            "Sealing Entry Points: Seal cracks and crevices where ants can enter your home.",
            "Baits: Use ant baits with insecticides to attract and eliminate ant colonies.",
            "Natural Predators: Encourage natural predators like spiders and parasitic wasps.",
            "Professional Pest Control: If the infestation is severe, consider hiring a pest control service."
        ]
    },
    {
        "name": "bees",
        "description": "Bees are flying insects known for their role in pollinating plants and producing honey. They are essential for agriculture and ecosystems. Bees live in hives with a queen, worker bees, and drones.",
        "management": [
            "Avoid Disturbing Hives: If you encounter a bee hive, avoid disturbing it, and contact a beekeeper or pest control professional for removal.",
            "Prevent Attracting Bees: Keep food and sugary drinks covered during outdoor activities.",
            "Plant Bee-Friendly Plants: Attract bees to your garden with native, pollinator-friendly plants.",
            "Seek Professional Assistance: If you have a bee infestation in or near your home, consult a beekeeper or pest control expert for safe removal."
        ]
    },
    {
        "name": "beetle",
        "description": "Beetles are insects belonging to the order Coleoptera, and they are among the most diverse and abundant organisms on Earth. They come in various shapes and sizes and can be herbivores, predators, or scavengers.",
        "management": [
            "Identification: Identify the specific beetle species to determine the most effective control methods.",
            "Cultural Practices: Maintain good garden hygiene, rotate crops, and remove debris to deter beetle infestations.",
            "Biological Control: Use natural predators like ladybugs or nematodes to manage beetle populations.",
            "Insecticides: Apply insecticides if the infestation is severe and other methods are ineffective."
        ]
    },
    {
        "name": "caterpillar",
        "description": "Caterpillars are the larval stage of butterflies and moths. They are typically herbivorous and can feed on various plants, causing damage to foliage.",
        "management": [
            "Handpicking: Physically remove caterpillars from plants and relocate them.",
            "Biological Control: Introduce natural predators like parasitic wasps or predatory beetles.",
            "Organic Sprays: Use organic insecticides like neem oil or Bacillus thuringiensis (BT) to control caterpillar infestations.",
            "Netting: Protect vulnerable plants with netting or row covers to prevent caterpillar access."
        ]
    },
    {
        "name": "earthworms",
        "description": "Earthworms are beneficial soil organisms that play a vital role in soil aeration and nutrient cycling. While they are usually helpful, they can occasionally surface during heavy rain.",
        "management": [
            "Non-Lethal Methods: If earthworms surface due to rain, gently return them to the soil.",
            "Improve Drainage: Ensure proper soil drainage to prevent earthworms from surfacing excessively."
        ]
    },
    {
        "name": "earwig",
        "description": "Earwigs are elongated insects with pincers on their abdomen. They are omnivorous and feed on a variety of organic matter, including plants.",
        "management": [
            "Traps: Set up traps using rolled-up newspapers or cardboard to attract and capture earwigs.",
            "Remove Hiding Places: Reduce hiding spots by cleaning up garden debris and mulch.",
            "Insecticides: Use insecticides sparingly if earwig infestations are severe."
        ]
    },
    {
        "name": "grasshopper",
        "description": "Grasshoppers are herbivorous insects known for their strong hind legs, which they use for jumping. They can consume large amounts of plant material.",
        "management": [
            "Barriers: Use physical barriers like row covers to protect plants from grasshoppers.",
            "Natural Predators: Attract birds, such as chickens or guinea fowl, which feed on grasshoppers.",
            "Insecticides: Apply insecticides when grasshopper populations reach damaging levels."
        ]
    },
    {
        "name": "moth",
        "description": "Moths are flying insects belonging to the order Lepidoptera. While most moths are harmless, certain species can be pests, particularly their larvae (caterpillars).",
        "management": [
            "Identification: Identify the specific moth or caterpillar species to determine appropriate control measures.",
            "Cultural Practices: Maintain good garden hygiene, remove caterpillars by hand, and encourage natural predators.",
            "Insecticides: Apply insecticides if necessary, using targeted products for moth or caterpillar control."
        ]
    },
    {
        "name": "slug",
        "description": "Slugs are soft-bodied, legless mollusks that feed on plant foliage. They are most active during damp and cool conditions.",
        "management": [
            "Physical Barriers: Place barriers like copper tape or diatomaceous earth around plants to deter slugs.",
            "Traps: Use beer traps or boards to attract and capture slugs.",
            "Natural Predators: Encourage natural slug predators like frogs, toads, and birds in your garden.",
            "Organic Baits: Apply organic slug baits containing iron phosphate."
        ]
    },
    {
        "name": "snail",
        "description": "Snails are similar to slugs but have a coiled shell. They are also mollusks that feed on plant foliage.",
        "management": [
            "Physical Barriers: Use barriers like copper tape or diatomaceous earth to keep snails away from plants.",
            "Handpicking: Collect snails by hand, especially during the evening or early morning when they are active.",
            "Natural Predators: Encourage natural predators like ducks, chickens, or beetles to control snail populations.",
            "Organic Baits: Apply organic snail baits containing iron phosphate."
        ]
    },
    {
        "name": "wasp",
        "description": "Wasps are flying insects, and many species are beneficial for pollination and pest control. However, some can be pests, particularly social wasps like yellowjackets.",
        "management": [
            "Prevent Nesting: Seal entry points to prevent wasps from building nests in or near your home.",
            "Traps: Use wasp traps to capture and reduce their numbers.",
            "Professional Removal: If a wasp nest is in a problematic location, consult a pest control professional for safe removal."
        ]
    },
    {
        "name": "weevil",
        "description": "Weevils are a type of beetle known for their elongated snouts. They are typically herbivorous and can infest stored grains, beans, and other food products.",
        "management": [
            "Storage Hygiene: Store grains and dry food products in airtight containers to prevent weevil infestations.",
            "Freezing: Freeze infested items for several days to kill weevil larvae.",
            "Discard Infested Food: Dispose of heavily infested food items."
        ]
    }
]

# return the info for the given name or the class name
def get_extras(name):
    for extra in PEST_INFO:
        if extra["name"] == name:
            return extra
    return None


# Download the blob from the bucket.
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name, destination_file_name))


# Prediction function
def pest_predict(request):
    global model
    global interpreter
    global input_index
    global output_index

    if model is None:
        download_blob(CLOUD_BUCKET_NAME,  "models/pest_detector.h5",
                      "/tmp/pest_detector.h5",)  # download the model

        model = tf.keras.models.load_model(
            "/tmp/pest_detector.h5")  # load the model

    image = request.files['file'].read()  # read the image

    image = Image.open(BytesIO(image)).convert("RGB")  # convert to RGB image
    image = np.array(image.resize((256, 256))) / 255.0  # normalize the image
    image_array = np.expand_dims(image, axis=0)  # add batch dimension

    prediction = model.predict(image_array)

    # get the info for the class name
    info = get_extras(CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "info": info
    }
