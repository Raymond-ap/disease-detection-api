from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "exp://172.20.10.3:19000"  # expo server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLASS_NAMES = ['Pepper bell Bacterial spot',
#                'Pepper bell healthy',
#                'Potato Early blight',
#                'Potato Late blight',
#                'Potato healthy',
#                'Tomato Bacterial spot',
#                'Tomato Early blight',
#                'Tomato Late blight',
#                'Tomato Leaf Mold',
#                'Tomato Septoria leaf spot',
#                'Tomato Target Spot',
#                'Tomato Tomato YellowLeaf Curl Virus',
#                'Tomato Tomato mosaic virus',
#                'Tomato healthy']


PLANT_CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

PLANT_MANAGEMENTS = [
    {
        "name": "Apple___Apple_scab",
        "symptoms": [
            "Circular, brown to black lesions on apple leaves, fruit, and stems.",
            "Lesions may have a dark border and cause fruit to shrivel and mummify.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Prune and remove infected plant parts.",
            "Apply fungicides during the growing season.",
            "Practice good orchard sanitation to reduce disease spread."
        ]
    },
    {
        "name": "Apple___Black_rot",
        "symptoms": [
            "Circular, brown to black lesions on apple leaves, fruit, and stems.",
            "Lesions may have a dark border and cause fruit to shrivel and mummify.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Prune and remove infected plant parts.",
            "Apply fungicides during the growing season.",
            "Practice good orchard sanitation to reduce disease spread."
        ]
    },
    {
        "name": "Apple___Cedar_apple_rust",
        "symptoms": [
            "Bright orange, spore-producing galls on apple leaves and fruit.",
            "Yellow or orange spots on apple leaves with small black dots.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Remove cedar trees nearby as they serve as alternate hosts.",
            "Apply fungicides during apple growing season.",
            "Prune and destroy infected plant parts."
        ]
    },
    {
        "name": "Apple___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Blueberry___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Cherry_(including_sour)___Powdery_mildew",
        "symptoms": [
            "White, powdery fungal growth on cherry leaves, fruit, and stems.",
            "Affected leaves may become distorted or curl.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Apply fungicides during the growing season.",
            "Prune and remove infected plant parts.",
            "Promote good air circulation in the orchard."
        ]
    },
    {
        "name": "Cherry_(including_sour)___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "symptoms": [
            "Small, circular to oval lesions with gray centers and dark borders on corn leaves.",
            "Leaves may turn yellow or brown, and lesions can coalesce.",
            "Reduced photosynthesis and yield loss."
        ],
        "treatments": [
            "Plant resistant corn varieties if available.",
            "Apply fungicides as a preventive measure.",
            "Maintain field sanitation by removing debris."
        ]
    },
    {
        "name": "Corn_(maize)___Common_rust_",
        "symptoms": [
            "Small, reddish-brown pustules (rusts) on corn leaves, stems, and husks.",
            "Rust pustules can rupture and release spores.",
            "Reduced photosynthesis, stunted growth, and yield loss."
        ],
        "treatments": [
            "Plant rust-resistant corn varieties if possible.",
            "Apply fungicides during the growing season as needed.",
            "Remove and destroy infected plant parts."
        ]
    },
    {
        "name": "Corn_(maize)___Northern_Leaf_Blight",
        "symptoms": [
            "Long, elliptical lesions with tan centers and dark borders on corn leaves.",
            "Lesions may coalesce and result in large, irregular areas of blighted tissue.",
            "Reduced photosynthesis and yield loss."
        ],
        "treatments": [
            "Plant resistant corn varieties if available.",
            "Apply fungicides during the growing season.",
            "Practice crop rotation and remove crop debris."
        ]
    },
    {
        "name": "Corn_(maize)___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Grape___Black_rot",
        "symptoms": [
            "Circular, brown lesions on leaves, fruit, and stems.",
            "Affected grapes become dark, shriveled, and mummified.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Prune and remove infected plant parts.",
            "Apply fungicides before and during the growing season.",
            "Promote good vineyard hygiene to reduce disease spread."
        ]
    },
    {
        "name": "Grape___Esca_(Black_Measles)",
        "symptoms": [
            "Black streaks or lesions on leaves, stems, and fruit.",
            "Affected leaves may have a reddish or purplish discoloration.",
            "Reduced vine growth and fruit quality."
        ],
        "treatments": [
            "Prune and remove infected plant parts.",
            "Apply fungicides as preventive measures.",
            "Improve vineyard sanitation."
        ]
    },
    {
        "name": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "symptoms": [
            "Small, circular to irregular spots on grape leaves.",
            "Spots may have a grayish-white center and dark margins.",
            "Premature leaf drop and reduced fruit quality."
        ],
        "treatments": [
            "Prune and remove infected leaves.",
            "Apply fungicides during the growing season.",
            "Maintain good air circulation."
        ]
    },
    {
        "name": "Grape___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Orange___Haunglongbing_(Citrus_greening)",
        "symptoms": [
            "Yellowing and blotchy mottling of leaves.",
            "Premature fruit drop and small, lopsided, bitter fruit.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Plant disease-free citrus stock.",
            "Control psyllid insects that spread the disease.",
            "Apply nutritional sprays to affected trees."
        ]
    },
    {
        "name": "Peach___Bacterial_spot",
        "symptoms": [
            "Circular, water-soaked lesions on peach leaves, often with a yellow halo.",
            "Lesions may also occur on fruit, causing raised, dark spots.",
            "Leaf drop and reduced fruit quality."
        ],
        "treatments": [
            "Apply copper-based sprays during the growing season.",
            "Remove and destroy infected plant parts.",
            "Practice good orchard sanitation to reduce disease spread."
        ]
    },
    {
        "name": "Peach___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Pepper,_bell___Bacterial_spot",
        "symptoms": [
            "Circular, water-soaked lesions on pepper leaves, turning dark with age.",
            "Lesions may have a yellow halo.",
            "Leaf distortion and fruit spots can occur."
        ],
        "treatments": [
            "Apply copper-based sprays as a preventive measure.",
            "Remove and destroy infected plant parts.",
            "Practice crop rotation to reduce disease pressure."
        ]
    },
    {
        "name": "Pepper,_bell___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Potato___Early_blight",
        "symptoms": [
            "Circular, dark brown lesions with concentric rings on potato leaves.",
            "Lesions may have a target-like appearance.",
            "Reduced plant vigor and yield."
        ],
        "treatments": [
            "Remove and destroy infected plant parts.",
            "Apply fungicides during the growing season.",
            "Practice crop rotation and avoid overhead watering."
        ]
    },
    {
        "name": "Potato___Late_blight",
        "symptoms": [
            "Dark, water-soaked lesions on potato leaves, often with white, fuzzy growth.",
            "Affected leaves may turn brown and die rapidly.",
            "Reduced yield and tuber quality."
        ],
        "treatments": [
            "Remove and destroy infected plant parts.",
            "Apply fungicides preventively, especially during wet conditions.",
            "Avoid planting infected seed potatoes."
        ]
    },
    {
        "name": "Potato___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Raspberry___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Soybean___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Squash___Powdery_mildew",
        "symptoms": [
            "White, powdery fungal growth on squash leaves, fruit, and stems.",
            "Leaves may become distorted or curl.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Apply fungicides during the growing season.",
            "Prune and remove infected plant parts.",
            "Promote good air circulation in the garden."
        ]
    },
    {
        "name": "Strawberry___Leaf_scorch",
        "symptoms": [
            "Margins of strawberry leaves turn brown and become scorched or necrotic.",
            "Leaves may curl or appear distorted.",
            "Reduced fruit production and overall plant vigor."
        ],
        "treatments": [
            "Ensure adequate irrigation to maintain soil moisture.",
            "Avoid over-fertilization, especially with high nitrogen levels.",
            "Remove and destroy affected leaves to prevent disease spread."
        ]
    },
    {
        "name": "Strawberry___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
    {
        "name": "Tomato___Bacterial_spot",
        "symptoms": [
            "Dark, raised lesions on tomato leaves, fruit, and stems.",
            "Lesions may ooze bacterial exudate.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Remove and destroy infected plant parts.",
            "Apply copper-based sprays.",
            "Practice crop rotation."
        ]
    },
    {
        "name": "Tomato___Early_blight",
        "symptoms": [
            "Dark, concentric rings on lower tomato leaves.",
            "Leaves may turn yellow and die, starting from the bottom of the plant.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Remove affected leaves and destroy them.",
            "Apply fungicides as a preventive measure.",
            "Ensure good airflow and spacing between plants."
        ]
    },
    {
        "name": "Tomato___Late_blight",
        "symptoms": [
            "Dark, water-soaked lesions on tomato leaves, stems, and fruit.",
            "White, fluffy spores may appear on the underside of leaves.",
            "Fruits can develop brown, irregular blotches."
        ],
        "treatments": [
            "Remove and destroy affected plant parts.",
            "Apply copper-based fungicides.",
            "Practice good garden hygiene to prevent spore spread."
        ]
    },
    {
        "name": "Tomato___Leaf_Mold",
        "symptoms": [
            "Fuzzy, white to yellow patches on tomato leaves, usually starting from older leaves.",
            "Affected leaves may curl and die."
        ],
        "treatments": [
            "Prune and remove affected leaves.",
            "Ensure good air circulation around tomato plants.",
            "Avoid overhead watering."
        ]
    },
    {
        "name": "Tomato___Septoria_leaf_spot",
        "symptoms": [
            "Small, circular spots with dark margins on tomato leaves.",
            "Spots may have a white center."
        ],
        "treatments": [
            "Remove and destroy affected leaves.",
            "Apply fungicides if necessary.",
            "Avoid overhead watering."
        ]
    },
    {
        "name": "Tomato___Spider_mites Two-spotted_spider_mite",
        "symptoms": [
            "Yellow stippling and tiny webs on the undersides of tomato leaves.",
            "Leaf discoloration and reduced plant vigor."
        ],
        "treatments": [
            "Spray plants with a strong stream of water to remove mites.",
            "Use miticides if infestations are severe.",
            "Increase humidity around plants."
        ]
    },
    {
        "name": "Tomato___Target_Spot",
        "symptoms": [
            "Dark, concentric rings on tomato leaves with a target-like appearance.",
            "Lesions may expand and cause defoliation.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Remove and destroy infected plant parts.",
            "Apply fungicides during the growing season.",
            "Ensure good air circulation in the garden."
        ]
    },
    {
        "name": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "symptoms": [
            "Yellowing and upward curling of tomato leaves.",
            "Stunted growth and reduced fruit yield."
        ],
        "treatments": [
            "Use virus-resistant tomato varieties.",
            "Manage whitefly populations, which transmit the virus."
        ]
    },
    {
        "name": "Tomato___Tomato_mosaic_virus",
        "symptoms": [
            "Mottled or streaked yellow and green patterns on tomato leaves.",
            "Reduced fruit yield and quality."
        ],
        "treatments": [
            "Remove and destroy infected plants.",
            "Control aphid populations, which transmit the virus.",
            "Use virus-free seedlings."
        ]
    },
    {
        "name": "Tomato___healthy",
        "symptoms": ["No specific symptoms or treatment needed."]
    },
]

PEST_CLASS_NAMES = ["ants", "bees", "beetle", "caterpillar", "earthworms", "earwig", "grasshopper",
                    "moth", "slug", "snail", "wasp", "weevil"]


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


# Definately not the best way to do this, but it works
PLANTMODEL = tf.keras.models.load_model("./models/plant_disease_classifier.h5")

PESTMODEL = tf.keras.models.load_model("./models/pest_detector.h5")


def get_plant_extras(name):
    for cure in PLANT_MANAGEMENTS:
        if cure["name"] == name:
            return cure
    return None


def get_extras(name):
    for extra in PEST_INFO:
        if extra["name"] == name:
            return extra
    return None


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((64, 64)))
    img = np.expand_dims(img, axis=0)
    prediction = PLANTMODEL.predict(img)
    confidence = float(np.max(prediction))
    info = get_plant_extras(PLANT_CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": PLANT_CLASS_NAMES[np.argmax(prediction)],
        "confidence": confidence,
        "info": info
    }


@app.post("/pest_predict")
async def pest_predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((256, 256)))
    img = np.expand_dims(img, axis=0)
    prediction = PESTMODEL.predict(img)

    info = get_extras(PEST_CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": PEST_CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "info": info
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
