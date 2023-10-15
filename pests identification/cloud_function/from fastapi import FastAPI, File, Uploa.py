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

CLASS_NAMES = ['Pepper bell Bacterial spot',
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
               'Tomato healthy']


# CLASS_NAMES = ['aphids',
#                'armyworm',
#                'beetle',
#                'bollworm',
#                'grasshopper',
#                'mites',
#                'mosquito',
#                'sawfly',
#                'stem_borer']



CLASS_NAMES = ["ants", "bees", "beetle", "caterpillar", "earthworms", "earwig", "grasshopper",
              "moth", "slug", "snail", "wasp", "weevil"]




# EXTRAS = [
#     {
#         "name": "aphids",
#         "description": "Aphids are small sap-sucking insects and members of the superfamily Aphidoidea. Common names include greenfly and blackfly, although individuals within a species can vary widely in color. The group includes the fluffy white woolly aphids",
#         "scientific_name": "Aphidoidea",
#         "kingdom": "Animalia",
#         "removal": [
#             "The safest and fastest method for how to get rid of aphids is to spray them off your plants with a strong stream of water from the garden hose. Aphids are such small, soft-bodied insects that even a good rainstorm can knock them off. Once aphids are knocked off a plant, they rarely climb back on",
#             " Insecticidal soaps and horticultural oil will kill aphids but must be applied regularly during heavy infestations since aphids reproduce so quickly",
#             "If you aren't the squeamish sort, and the infestation isn't so heavy that it would take forever to clean off, a non-toxic process of how to get rid of aphids is to gently rub your thumb and fingertips over your plants' leaves and stems wherever you see them",
#         ],
#         "preventions": [
#             "Scout for Aphids Regularly",
#             "Use Row Covers in Your Vegetable Garden",
#             "Remove Weeds",
#             "Plant flowers, including marigolds, calendula, sunflower, daisy, alyssum, or dill nearby to attract beneficial insects that love to feed on aphids. Ladybugs and lacewings are especially effective at devouring them",
#         ],
#         "characteristics": [
#             "Very small eyes",
#             "Sucking mouthparts in the form of a relatively long",
#             "Segmented rostrum, and fairly long antennae"
#         ]
#     },
#     {
#         "name": "sawfly",
#         "description": "Sawflies are the insects of the suborder Symphyta within the order Hymenoptera, alongside ants, bees, and wasps. The common name comes from the saw-like appearance of the ovipositor, which the females use to cut into the plants where they lay their eggs",
#         "scientific_name": "Symphyta",
#         "kingdom": "Animalia",
#         "removal": [
#             "If you have a small number of rose slugs on just a few plants, the best approach would be to hand-pick them off and drop them in a cup of soapy water",
#             "You can also use a forceful spray of water out of a garden hose, which will knock off and destroy many of the larvae",
#             "Be sure to spray the water on the leaves' upper and undersides",
#             "Using insecticidal soap or neem oil to control these pests is also an option",
#         ],
#         "characteristics": [
#             "They look like fat-bodied flies without the pinched waist that is characteristic of the better-known wasps",
#             "Sawflies have four wings, while all of the true flies have only two",
#             "Sawfly wasps cannot sting. Sawfly larvae look like hairless caterpillars"
#         ]
#     },
#     {
#         "name": "stem_borer",
#         "description": "It is found in aquatic environments where there is continuous flooding. Second instar larvae enclose themselves in body leaf wrappings to make tubes and detach themselves from the leaf and falls onto the water surface. They attach themselves to the tiller and bore into the stem",
#         "preventions": [
#             "Avoid close planting and continuous water stagnation at early stages",
#             "Collect and destroy the egg masses",
#             "Pull out and destroy the affected tillers",
#             "Removal and proper disposal of stubbles will keep the borer population low in next crop"
#         ],
#         "kingdom": "Animalia",
#         "removal": [
#             "ETL : 2 egg masses/m2 (or) 10% dead heart at vegetative stage (or) 2% White ear at Flowering stage",
#             "When natural enemies of stem borers are present, application of chemical measures can be delayed or dispensed with",
#             "When natural enemies of stem borers are present, application of chemical measures can be delayed or dispensed with",
#             "Spray any one of the following based on ETLs :Quinalphos 25 EC 1000 ml/ha (or) Phosphamidon 40 SL 600 ml/ha (or) Profenophos 50 EC 1000 ml/ha",
#         ],
#         "characteristics": [
#             "Has bright yellowish brown with a black spot at the centre of the fore wings and a tuft of yellow hairs at the anal region",
#             "Smaller with pale yellow forewings without black spot",
#         ]
#     },
#     {
#         "name": "mosquito",
#         "description": "Mosquitoes are approximately 3,600 species of small flies comprising the family Culicidae. The word 'mosquito' is Spanish for 'little fly'. Mosquitoes have a slender segmented body, one pair of wings, one pair of halteres, three pairs of long hair-like legs, and elongated mouthparts",
#         "preventions": [
#             'Use of parasites, predator (fish & frog) and pathogens. The best known fish are Gambusia affinis and lobister reteculatus',
#             "Release of sterile male mosquitoes in the field. Which will compete with the natural fertile male mosquito in mating & the population will be automatically reduced",
#             "To eliminate their breeding places. This is also known is source reduction. Filling, leveling and drainage of breeding places"
#         ],
#         "scientific_name": "Culicidae",
#         "kingdom": "Animalia",
#         "removal": [
#             "Anti larval chemicals like Paris green, mineral oil",
#             "Anti adult chemicals like Melathian, Fenthian and abate"
#         ],
#         "characteristics": [
#             "Mosquitoes have two large compound eyes that detect movement",
#             "Organs between the antennae that sense odor"
#         ]
#     },
#     {
#         "name": "grasshopper",
#         "description": "Grasshoppers are a group of insects belonging to the suborder Caelifera. They are among what is possibly the most ancient living group of chewing herbivorous insects, dating back to the early Triassic around 250 million years ago",
#         "kingdom": "Animalia",
#         "removal": [
#             "Poultry, including chickens and guinea hens, are excellent predators but can also cause damage to some garden plants"
#             " Cones, screened boxes, floating row covers, and other protective covers provide some protection if the number of pests isn't high",
#             'They can be handpicked and squashed',
#         ],
#     },
#     {
#         "name": "mites",
#         "description": "Mites are small arachnids. Mites span two large orders of arachnids, the Acariformes and the Parasitiformes, which were historically grouped together in the subclass Acari",
#         "preventions": [
#             "Contact pesticides: These are typically applied in liquid form as a fine spray. This kills mites quickly before they have time to multiply"
#             "Systemic pesticides: You can apply these types of pesticides in several different ways. They can be mixed with water and poured over the soil, sprinkled over the plant as a granular product, or pushed into the soil as a “pin” or “spike” as well as being sprayed onto the plant. The active ingredients in the pesticide enter the plant and poison pests after they feed on the plant"
#         ],
#         "scientific_name": "Arachnida",
#         "kingdom": "Animalia",
#         "characteristics": [
#             "Mites have bulbous, round, or pill-shaped bodies",
#             "Their size varies by species, but most mites are usually invisible to the naked eye. The largest mites measure about 6 mm long, while the smallest are about 0.1 mm"
#         ]
#     },
#     {
#         "name": "armyworm",
#         "description": "The fall armyworm is a species in the order Lepidoptera and one of the species of the fall armyworm moths distinguished by their larval life stage. The term 'armyworm' can refer to several species, often describing the large-scale invasive behavior of the species' larval stage",
#         "scientific_name": "Spodoptera frugiperda",
#         "kingdom": "Animalia",
#         "preventions": [
#             "While scouting for damage is important for all insects, careful inspection is especially important for this pest due to the rapid nature of its destructive feeding.  If armyworms are present in turfgrasses in large numbers, it is important to treat as soon as possible to avoid further injury. There are several active ingredients that are effective in controlling fall armyworms, but many variations exist in formulation, use site, applicator requirements, etc.  As always, be sure to follow the product label for specific instructions on timing, use rate, and application methods. For a complete list of products labeled for fall armyworm control, consult the Texas Turfgrass Pest Control Recommendations Guide"
#         ],
#     },
#     {
#         "name": "bollworm",
#         "description": "The African bollworm is a pest of major importance in most areas where it occurs. It damages a wide variety of food, fibre, oilseed, fodder and horticultural crops. It is a major pest due to its high mobility, its ability to feed on many species of plants, its high fecundity and reproductive rate, and its capacity to develop resistance to pesticides. The habit of feeding inside the fruiting parts of the plant during most of its development makes bollworms less vulnerable to insecticides",
#         "kingdom": "Animalia",
#         "preventions": [
#             "Remove and destroy plant residues immediately after harvesting",
#             "Plough the soil after harvesting. This exposes pupae, which may then be killed by natural enemies or through desiccation by the sun"
#         ],
#         "removal": [
#             "Obtain a suitable trap crop to plant with the main crop",
#             "Plant the trap crop around the vegetable field in strips 10 to 15 cm apart; pigeon peas can be planted as a hedge around the main crop",
#             "Plant the trap crop so that it starts flowering earlier than the main crop and remains flowering thorough the development cycle of the main crop. This way, the bollworms will lay eggs and thrive only on the trap crops",
#             "Regularly observe the populations of bollworms on the trap crop and, if necessary, spray them with a suitable pesticide to control them"
#         ],
#         "characteristics": []
#     },
#     {
#         "name": "beetle",
#         "description": "Beetles are insects that form the order Coleoptera, in the superorder Endopterygota. Their front pair of wings are hardened into wing-cases, elytra, distinguishing them from most other insects",
#         "scientific_name": "Coleoptera",
#         "Kingdom": "Animalia",
#         "removal": [
#             "Use water and dish soap. While this is a manual approach, it can be effective",
#             "Vacuum beetles up. Using a wet/dry or ShopVac, suck beetles up where you see them resting or moving",
#             "Use insecticidal soap on bushes and landscaping",
#             "For a hands-off way to get rid of beetles in your yard, hang traps around your lawn. You can purchase these traps at most home improvement stores in your area",
#         ],
#         "characteristics": [
#             "Invertebrate"
#             "Three pairs of jointed legs"
#             "Protective exoskeleton"
#             "Three-part body (head, thorax and abdomen)"
#             "Single pair of compound eyes (some beetle species have additional simple eyes “ocelli” on the back of the head)"
#             "Wing-cases (modified forewings)"
#             "Either a single pair of wings, or no wings"
#         ]
#     }
# ]



# Definately not the best way to do this, but it works
CURES = [
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

PLANTMODEL = tf.keras.models.load_model("./models/version_3.h5")

# PESTMODEL = tf.keras.models.load_model("./models/pest_version_1.h5")
PESTMODEL = tf.keras.models.load_model("./models/pest_detector.h5")


# return the info for the given name or the class name
def get_cure(name):
    for cure in CURES:
        if cure["name"] == name:
            return cure
    return None


def get_extras(name):
    for extra in EXTRAS:
        if extra["name"] == name:
            return extra
    return None


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((256, 256))) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = PLANTMODEL.predict(img)
    info = get_cure(CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "info": info
    }


@app.post("/pest_predict")
async def pest_predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((256, 256))) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = PESTMODEL.predict(img)

    info = get_extras(CLASS_NAMES[np.argmax(prediction)])

    return {
        "class_name": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "info": info
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
