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
    'Pepper bell Bacterial spot',
    'Pepper bell healthy',
    'Potato Early blight',
    'Potato Late blight',
    'Potato healthy',
    'Tomato Bacterial spot',
    'Tomato Early blight',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Target Spot',
    'Tomato Tomato YellowLeaf Curl Virus',
    'Tomato Tomato mosaic virus',
    'Tomato healthy'
]

PLANT_MANAGEMENTS = [
    {
        "name": "Pepper bell Bacterial spot",
        "symptoms":
            "Disease symptoms can appear throughout the above-ground portion of the plant, which may include leaf spot, fruit spot and stem canker. However, early symptoms show up as water-soaked lesions on leaves that can quickly change from green to dark brown and enlarge into spots that are up to 1/4 inch in diameter with slightly raised margins. Over time, these spots can dry up in less humid weather, which allows the damaged tissues to fall off, resulting in a tattered appearance on the affected leaves",
        "treatments": [
            "Washing seeds for 40 minutes in diluted Clorox (two parts Clorox plus eight parts water) is effective in reducing the bacterial population on a seed’s surface. However, bacteria inside the seeds are little affected by this treatment. ",
            "Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C, is effective in reducing bacterial populations on the surface and inside the seeds. However, seed germination may be affected by heat treatment if not done accurately, while the risk is relatively low with bleach treatment. ",
            "Control of bacterial spot on greenhouse transplants is an essential step for preventing the spread of the leaf spot bacteria in the field. Transplants should be inspected regularly to identify symptomatic seedlings. Transplants with symptoms may be removed and destroyed or treated with streptomycin, if detected at the very early stage of disease development. It should be noted that strains of leaf spot bacteria resistant to streptomycin may arise with multiple applications of streptomycin. ",
            "Good cultural practices include avoiding all conditions that enable the pathogen to spread and multiply rapidly. Bacteria spread with splashing water. Therefore, overhead irrigation method should be replaced with drip irrigation and the field should not be accessed when plants are wet. ",
        ]
    },
    {
        "name": "Potato Early blight",
        "symptoms":
            "The first symptoms of early blight appear as small, circular or irregular, dark-brown to black spots on the older (lower) leaves. These spots enlarge up to 3/8 inch in diameter and gradually may become angular-shaped.",
        "treatments": [
            "Avoid nitrogen and phosphorus deficiency.",
            "Select a late-season variety with a lower susceptibility to early blight. (Resistance is associated with plant maturity and early maturing cultivars are more susceptible).",
            "Time irrigation to minimize leaf wetness duration during cloudy weather and allow sufficient time for leaves to dry prior to nightfall.",
            "Rotate foliar fungicides to prevent the development of fungicide resistance.",
            "Eradicate weed hosts such as hairy nightshade to reduce inoculum for future plantings.",
            "Rotate fields to non-host crops for at least three years (three to four-year crop rotation).",
        ]
    },
    {
        "name": "Potato Late blight",
        "symptoms":
            "The first symptoms of late blight in the field are small, light to dark green, circular to irregular-shaped water-soaked spots. These lesions usually appear first on the lower leaves. Lesions often begin to develop near the leaf tips or edges, where dew is retained the longest. During cool, moist weather, these lesions expand rapidly into large, dark brown or black lesions, often appearing greasy. Leaf lesions also frequently are surrounded by a yellow chlorotic halo",
        "treatments": [
            "Avoid planting problem areas that may remain wet for extended periods or may be difficult to spray (the field near the center of the pivot, along powerlines and tree lines).",
            "Destroy all cull and volunteer potatoes.",
            "Plant late blight-free seed tubers.",
            "Do not mix seed lots because cutting can transmit late blight.",
            "Avoid excessive and/or nighttime irrigation.",
            "Eliminate sources of inoculum such as hairy nightshade weed species and volunteer potatoes.",
            "Applying phosphorous acid to potatoes after harvest and before piling can prevent infection and the spread of late blight in storage.",
            "Monitor home garden and market tomatoes near you for late blight. Late blight can move from these local sources to potato fields.",
        ]
    },
    {
        "name": "Tomato Bacterial spot",
        "symptoms":
        "When it first appears on the leaves, bacterial spot is similar in appearance to many other tomatoes diseases. Tomato leaves have small (less than 1/8 inch), brown, circular spots surrounded by a yellow halo. The center of the leaf spots often falls out resulting in small holes. The leaf spots do not contain concentric rings, spots with concentric rings are likely caused by early blight.",
        "treatments": [
            "Look for leaves with spots, especially during periods of wet, humid weather. Remove and destroy infected leaves. ",
            "There are many varieties of bell pepper and hot pepper with resistance to bacterial spot.",
            "A few tomato varieties with resistance are available."
            "Plant tomatoes where no tomatoes, potatoes, peppers or eggplants have been for the past 3-4 years.",
            "Keep tomato leaves as dry as possible. Water with drip irrigation or a soaker hose. If watering by hand, do so at the base of the plant.",
            "Space plants so that air flows between them.  This could range from 18-24 inches for caged tomatoes to 30-36 inches for uncaged determinate tomatoes."
        ]
    },
    {
        "name": "Tomato Early blight",
        "symptoms":
            "Initially, small dark spots form on older foliage near the ground. Leaf spots are round, brown and can grow up to 1/2 inch in diameter. Larger spots have target-like concentric rings. The tissue around spots often turns yellow. Severely infected leaves turn brown and fall off, or dead, dried leaves may cling to the stem. Seedling stems are infected at or just above the soil line. The stem turns brown, sunken and dry (collar rot). If the infection girdles the stem, the seedling wilts and dies. Fruit can be infected at any stage of maturity. Fruit spots are leathery and black, with raised concentric ridges. They generally occur near the stem. Infected fruit may drop from the plant.",
        "treatments": [
            "Cover the soil under the plants with mulch, such as fabric, straw, plastic mulch, or dried leaves.",
            "Water at the base of each plant, using drip irrigation, a soaker hose, or careful hand watering.",
            "Increase airflow by staking or trellising, removing weeds, and spacing plants adequately apart",
            "Pruning the bottom leaves can also prev",
            "Pinch off leaves with leaf spots and bury them in the compost pile.",
            "Let two years pass before you plant tomatoes or peppers in the same location",
            "Early blight-resistant varieties are readily available.  As early blight occurs commonly in Minnesota, gardeners should look into these varieties.",
            "Resistance does not mean you will not see any early blight; rather, resistant varieties can better tolerate the pathogens, and so the damage will be less severe than with non-resistant varieties."
        ]
    },
    {
        "name": "Tomato Late blight",
        "symptoms":
            "Leaves have large, dark brown blotches with a green gray edge; not confined by major leaf veins. Infections progress through leaflets and petioles, resulting in large sections of dry brown foliage. In high humidity, thin powdery white fungal growth appears on infected leaves, tomato fruit and stems. Infected potato tubers become discolored (anywhere from brown to red to purple), and infected by secondary soft rot bacteria", "treatments": [
                "Plant tomatoes where no tomatoes, potatoes, peppers or eggplants have been for the past 3-4 years.",
                "Plant tomatoes where no tomatoes, potatoes, peppers or eggplants have been for the past 3-4 years.",
                "Remove or bury plants at the end of the season. Manage cull piles so culls break down over winter.",
                "Late blight isn’t often seen in Minnesota gardens, and pesticides sprays are not recommended in the home garden."
            ]
    },
    {
        "name": "Tomato Leaf Mold",
        "symptoms": "",
        "treatments": [
            "Use drip irrigation and avoid watering foliage.",
            "Space plants to provide good air movement between rows and individual plants.",
            "Stake, string or prune to increase airflow in and around the plant.",
            "Sterilize stakes, ties, trellises, etc. with 10percent household bleach or commercial sanitizer.",
            "Keep night temperatures in greenhouses higher than outside temperatures to avoid dew formation on the foliage.",
            "Clean the high tunnel or greenhouse walls and benches at the end of the season with a commercial sanitizer.",
            "Fungicide applications should be made prior to infection when environmental conditions favor disease to be the most effective.",
        ]
    },
    {
        "name": "Tomato Septoria leaf spot",
        "symptoms": "Septoria leaf spot starts on lower leaves as small, circular gray lesions (spots) with dark borders. Fungal lesions enlarge, coalesce, and cause leaves to yellow and die. Lesions usually appear when the first fruit begins to form. Tiny black pycnidia (fungal fruiting bodies) can be seen in the lesions. Favored by wet weather.",
        "treatments": [
            "Provide adequate spacing to increase air circulation and remove all suckers that emerge from the plant base",
            "Monitor transplants carefully for signs of this disease.",
            "Keep plants well mulched to minimize soil splashing.",
            "Water plants at their base. Avoid wetting the foliage.",
            "Prune off the lowest 3-4 leaf branches once plants are well established and starting to develop fruits.",
            "Remove infected leaves during the growing season and remove all infected plant parts at the end of the season.",
        ]
    },
    {
        "name": "Tomato Target Spot",
        "symptoms": "The target spot fungus can infect all above-ground parts of the tomato plant. Plants are most susceptible as seedlings and just before and during fruiting. The initial foliar symptoms are pinpoint-sized, water-soaked spots on the upper leaf surface. The spots develop into small, necrotic lesions that have light brown centers and dark margins. These symptoms may be confused with symptoms of bacterial spot.",
        "treatments": [
            "Cultural practices for target spot management include improving airflow through the canopy by wider plant spacing and avoiding over-fertilizing with nitrogen, which can cause overly lush canopy formation.",
            "Pruning suckers and older leaves in the lower canopy can also increase airflow and reduce leaf wetness.",
            "Avoid planting tomatoes near old plantings. Inspect seedlings for target spot symptoms before transplanting."
        ]
    },
    {
        "name": "Tomato Tomato YellowLeaf Curl Virus",
        "symptoms":
        "The most obvious symptoms in tomato plants are small leaves that become yellow between the veins. The leaves also curl upwards and towards the middle of the leaf",
        "treatments": [
            "Inspect plants for whitefly infestations two times per week. If whiteflies are beginning to appear, spray with azadirachtin (Neem), pyrethrin or insecticidal soap. For more effective control, it is recommended that at least two of the above insecticides be rotated at each spraying. Follow label directions closely for dosage rates, spray intervals and precautions. Spray the undersides of the leaves thoroughly.",
            "If symptomatic plants have no obvious whiteflies on the lower leaf surface, these plants can be cut from the garden and BURIED in the compost."
        ]
    },
    {
        "name": "Tomato Tomato mosaic virus",
        "symptoms":
        "The foliage of affected tomato plants shows mottling, with alternating yellowish and darker green areas, the latter often appearing thicker and raised giving a blister-like appearance. The leaves tend to be fern-like in appearance with pointed tips and younger leaves may be twisted. The fruit may be distorted, yellow blotches and necrotic spots may occur on both ripe and green fruit and there may be internal browning of the fruit wall. In young plants, the infection reduces the set of fruit and may cause distortions and blemishes.",
        "treatments": [
            "Monitor the rest of your plants closely, especially those that were located near infected plants.",
            "Disinfect gardening tools after every use. Keep a bottle of a weak bleach solution or other antiviral disinfectant to wipe your tools down with.",
            "Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. Burn infected plants or throw them out with the garbage."
        ]
    }
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
PLANTMODEL = tf.keras.models.load_model("./models/version_3.h5")

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
    img = np.array(img.resize((256, 256))) / 256
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
