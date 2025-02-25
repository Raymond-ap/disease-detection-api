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

CLASS_NAMES = ['aphids',
               'armyworm',
               'beetle',
               'bollworm',
               'grasshopper',
               'mites',
               'mosquito',
               'sawfly',
               'stem_borer']


EXTRAS = [
    {
        "name": "aphids",
        "description": "Aphids are small sap-sucking insects and members of the superfamily Aphidoidea. Common names include greenfly and blackfly, although individuals within a species can vary widely in color. The group includes the fluffy white woolly aphids",
        "scientific_name": "Aphidoidea",
        "kingdom": "Animalia",
        "removal": [
            "The safest and fastest method for how to get rid of aphids is to spray them off your plants with a strong stream of water from the garden hose. Aphids are such small, soft-bodied insects that even a good rainstorm can knock them off. Once aphids are knocked off a plant, they rarely climb back on",
            " Insecticidal soaps and horticultural oil will kill aphids but must be applied regularly during heavy infestations since aphids reproduce so quickly",
            "If you aren't the squeamish sort, and the infestation isn't so heavy that it would take forever to clean off, a non-toxic process of how to get rid of aphids is to gently rub your thumb and fingertips over your plants' leaves and stems wherever you see them",
        ],
        "preventions": [
            "Scout for Aphids Regularly",
            "Use Row Covers in Your Vegetable Garden",
            "Remove Weeds",
            "Plant flowers, including marigolds, calendula, sunflower, daisy, alyssum, or dill nearby to attract beneficial insects that love to feed on aphids. Ladybugs and lacewings are especially effective at devouring them",
        ],
        "characteristics": [
            "Very small eyes",
            "Sucking mouthparts in the form of a relatively long",
            "Segmented rostrum, and fairly long antennae"
        ]
    },
    {
        "name": "sawfly",
        "description": "Sawflies are the insects of the suborder Symphyta within the order Hymenoptera, alongside ants, bees, and wasps. The common name comes from the saw-like appearance of the ovipositor, which the females use to cut into the plants where they lay their eggs",
        "scientific_name": "Symphyta",
        "kingdom": "Animalia",
        "removal": [
            "If you have a small number of rose slugs on just a few plants, the best approach would be to hand-pick them off and drop them in a cup of soapy water",
            "You can also use a forceful spray of water out of a garden hose, which will knock off and destroy many of the larvae",
            "Be sure to spray the water on the leaves' upper and undersides",
            "Using insecticidal soap or neem oil to control these pests is also an option",
        ],
        "characteristics": [
            "They look like fat-bodied flies without the pinched waist that is characteristic of the better-known wasps",
            "Sawflies have four wings, while all of the true flies have only two",
            "Sawfly wasps cannot sting. Sawfly larvae look like hairless caterpillars"
        ]
    },
    {
        "name": "stem_borer",
        "description": "It is found in aquatic environments where there is continuous flooding. Second instar larvae enclose themselves in body leaf wrappings to make tubes and detach themselves from the leaf and falls onto the water surface. They attach themselves to the tiller and bore into the stem",
        "preventions": [
            "Avoid close planting and continuous water stagnation at early stages",
            "Collect and destroy the egg masses",
            "Pull out and destroy the affected tillers",
            "Removal and proper disposal of stubbles will keep the borer population low in next crop"
        ],
        "kingdom": "Animalia",
        "removal": [
            "ETL : 2 egg masses/m2 (or) 10% dead heart at vegetative stage (or) 2% White ear at Flowering stage",
            "When natural enemies of stem borers are present, application of chemical measures can be delayed or dispensed with",
            "When natural enemies of stem borers are present, application of chemical measures can be delayed or dispensed with",
            "Spray any one of the following based on ETLs :Quinalphos 25 EC 1000 ml/ha (or) Phosphamidon 40 SL 600 ml/ha (or) Profenophos 50 EC 1000 ml/ha",
        ],
        "characteristics": [
            "Has bright yellowish brown with a black spot at the centre of the fore wings and a tuft of yellow hairs at the anal region",
            "Smaller with pale yellow forewings without black spot",
        ]
    },
    {
        "name": "mosquito",
        "description": "Mosquitoes are approximately 3,600 species of small flies comprising the family Culicidae. The word 'mosquito' is Spanish for 'little fly'. Mosquitoes have a slender segmented body, one pair of wings, one pair of halteres, three pairs of long hair-like legs, and elongated mouthparts",
        "preventions": [
            'Use of parasites, predator (fish & frog) and pathogens. The best known fish are Gambusia affinis and lobister reteculatus',
            "Release of sterile male mosquitoes in the field. Which will compete with the natural fertile male mosquito in mating & the population will be automatically reduced",
            "To eliminate their breeding places. This is also known is source reduction. Filling, leveling and drainage of breeding places"
        ],
        "scientific_name": "Culicidae",
        "kingdom": "Animalia",
        "removal": [
            "Anti larval chemicals like Paris green, mineral oil",
            "Anti adult chemicals like Melathian, Fenthian and abate"
        ],
        "characteristics": [
            "Mosquitoes have two large compound eyes that detect movement",
            "Organs between the antennae that sense odor"
        ]
    },
    {
        "name": "grasshopper",
        "description": "Grasshoppers are a group of insects belonging to the suborder Caelifera. They are among what is possibly the most ancient living group of chewing herbivorous insects, dating back to the early Triassic around 250 million years ago",
        "kingdom": "Animalia",
        "removal": [
            "Poultry, including chickens and guinea hens, are excellent predators but can also cause damage to some garden plants"
            " Cones, screened boxes, floating row covers, and other protective covers provide some protection if the number of pests isn't high",
            'They can be handpicked and squashed',
        ],
    },
    {
        "name": "mites",
        "description": "Mites are small arachnids. Mites span two large orders of arachnids, the Acariformes and the Parasitiformes, which were historically grouped together in the subclass Acari",
        "preventions": [
            "Contact pesticides: These are typically applied in liquid form as a fine spray. This kills mites quickly before they have time to multiply"
            "Systemic pesticides: You can apply these types of pesticides in several different ways. They can be mixed with water and poured over the soil, sprinkled over the plant as a granular product, or pushed into the soil as a “pin” or “spike” as well as being sprayed onto the plant. The active ingredients in the pesticide enter the plant and poison pests after they feed on the plant"
        ],
        "scientific_name": "Arachnida",
        "kingdom": "Animalia",
        "characteristics": [
            "Mites have bulbous, round, or pill-shaped bodies",
            "Their size varies by species, but most mites are usually invisible to the naked eye. The largest mites measure about 6 mm long, while the smallest are about 0.1 mm"
        ]
    },
    {
        "name": "armyworm",
        "description": "The fall armyworm is a species in the order Lepidoptera and one of the species of the fall armyworm moths distinguished by their larval life stage. The term 'armyworm' can refer to several species, often describing the large-scale invasive behavior of the species' larval stage",
        "scientific_name": "Spodoptera frugiperda",
        "kingdom": "Animalia",
        "preventions": [
            "While scouting for damage is important for all insects, careful inspection is especially important for this pest due to the rapid nature of its destructive feeding.  If armyworms are present in turfgrasses in large numbers, it is important to treat as soon as possible to avoid further injury. There are several active ingredients that are effective in controlling fall armyworms, but many variations exist in formulation, use site, applicator requirements, etc.  As always, be sure to follow the product label for specific instructions on timing, use rate, and application methods. For a complete list of products labeled for fall armyworm control, consult the Texas Turfgrass Pest Control Recommendations Guide"
        ],
    },
    {
        "name": "bollworm",
        "description": "The African bollworm is a pest of major importance in most areas where it occurs. It damages a wide variety of food, fibre, oilseed, fodder and horticultural crops. It is a major pest due to its high mobility, its ability to feed on many species of plants, its high fecundity and reproductive rate, and its capacity to develop resistance to pesticides. The habit of feeding inside the fruiting parts of the plant during most of its development makes bollworms less vulnerable to insecticides",
        "kingdom": "Animalia",
        "preventions": [
            "Remove and destroy plant residues immediately after harvesting",
            "Plough the soil after harvesting. This exposes pupae, which may then be killed by natural enemies or through desiccation by the sun"
        ],
        "removal": [
            "Obtain a suitable trap crop to plant with the main crop",
            "Plant the trap crop around the vegetable field in strips 10 to 15 cm apart; pigeon peas can be planted as a hedge around the main crop",
            "Plant the trap crop so that it starts flowering earlier than the main crop and remains flowering thorough the development cycle of the main crop. This way, the bollworms will lay eggs and thrive only on the trap crops",
            "Regularly observe the populations of bollworms on the trap crop and, if necessary, spray them with a suitable pesticide to control them"
        ],
        "characteristics": []
    },
    {
        "name": "beetle",
        "description": "Beetles are insects that form the order Coleoptera, in the superorder Endopterygota. Their front pair of wings are hardened into wing-cases, elytra, distinguishing them from most other insects",
        "scientific_name": "Coleoptera",
        "Kingdom": "Animalia",
        "removal": [
            "Use water and dish soap. While this is a manual approach, it can be effective",
            "Vacuum beetles up. Using a wet/dry or ShopVac, suck beetles up where you see them resting or moving",
            "Use insecticidal soap on bushes and landscaping",
            "For a hands-off way to get rid of beetles in your yard, hang traps around your lawn. You can purchase these traps at most home improvement stores in your area",
        ],
        "characteristics": [
            "Invertebrate"
            "Three pairs of jointed legs"
            "Protective exoskeleton"
            "Three-part body (head, thorax and abdomen)"
            "Single pair of compound eyes (some beetle species have additional simple eyes “ocelli” on the back of the head)"
            "Wing-cases (modified forewings)"
            "Either a single pair of wings, or no wings"
        ]
    }
]


# return the info for the given name or the class name
def get_extras(name):
    for extra in EXTRAS:
        if extra["name"] == name:
            return extra
    return None

MODEL = tf.keras.models.load_model("./models/pest_version_1.h5")


@app.post("/pest_predict")
async def pest_predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((256, 256))) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = MODEL.predict(img)

    info = get_extras(CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "info": info
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
