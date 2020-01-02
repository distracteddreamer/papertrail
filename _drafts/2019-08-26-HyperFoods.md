---
layout: post
title:  "HyperFoods"
date:   2019-08-26 7:51:32 +0100
categories: jekyll update
---
## The problem

Scale of the problem
> With rapidly ageing populations, the world is experiencing an unsustainable healthcare and economic burden from chronic diseases such as cancer, cardiovascular, metabolic and **neurodegenerative disorders**. [Emphasis added]

Scope of the solution
 Diet and nutritional factors play an essential role in the prevention of these diseases and significantly influence disease outcome in patients during and after therapy.

They identify the benefits of plant-based foods for cancer prevention and treatment. What are similar properties that will be beneficial for neurodegenerative diseases.

Many factors from how the food is cultivated, processed, prepared, cooked, combined with other foods influence what the active compounds are present and result in a vast search space which cannot be explored by traditional experimental methods.

Key types of work done in this area:
- Identification of food molecules that are similar to drugs based on their structure or individual gene-encoding protein targets.
- Using *"-omics"* data to extract insights on positive and / or adverse interactions between foods, drugs and disease.
- Using chemo-informatics and NLP to define health-promoting or detrimental links between foods and disease phenotypes from PubMed abstracts.

## Methods
Complex diseases like cancer involve a breakdown of molecular functions mediated through networks or molecular interactions. They used graph networks to simulate the downstream influence of therapeutics on human proteome networks. 

Data consists of interaction between drugs and their protein / gene targets. Most drugs work by binding to a specific subset of proteins. There is also a gene-gene connection network. These are use to create propagated molecule profiles for different drugs and foods. These profiles are then classified as potential anti-cancer candidates. 

Data available on two types of interactions:
1. Molecule to gene-encoded protein interactions - sparse data (presumably each of many molecules acts on just a few out of many proteins)
2. Protein-protein interactions
- Drug-protein interactions are mapped onto the whole set of protein-protein interactions in humans i.e. interactome networks - what is meant by "mapped" here?
- Mapped
    - Most drugs work by binding to a subset of proteins
    -  Proteins in turn function mostly as part of highly interconnected network
- Drugs "perturb" the human proteome network
    - Simulated using random walks on graphs with restarts controlled by network diffusion parameter "c"
    - Presumably the data is used to get transition probabilities e.g. protein given drug then protein given protein
    - How it works
        - For a given molecule / drug there is a small number of proteins that it targets
        - Other proteins are assigned scores based on their network proximity to the targets - is "network proximity" based on the protein-protein interaction network
        - Results in a genome-wide "profile" for a drug

    - Classification
        - The drug profile I think becomes the input to a classifier - not sure how features are extracted from it
        - Not sure what is meant when they say that the "best obtained models were used to predict the probability of a given existing approved drug to exhibit anti-cancer properties" - do they mean predicting whether a drug that is not used for cancer treatment would work for this purpose?
        - What is drug repositioning?
        - How was model validated for anti-cancer drug repositioning? By comparing with the other methodologies? (Which require additional datasets whose availability for food molecules is limited)
        - Small number of example and large number of features make linear classifiers preferrable.
        - Log-transformation of profiles (how was this done?) improved performance of classifiers
        - Reason this might be is that in log space, individual isolated genes which don't propagate (what does that mean?) and thus stay with a very high perturbation level (what does that mean?) would have lesser effect on the overall profile. 
        - "c" parameter had less pronounced effects (less than what?)
        - Different matching settings between compounds and genes also had less pronounced effects (again less than what)
        - Also gene-gene connection thresholds didn't have a strong influence - in general
        - They suggest the reason that these factors did not have strong influence is that connections in STRING (what is this?) include a wide range of knowledge sources - so the gene-gene (or protein-protein) interaction graph is quite complete and representative - as there is a large number of connections, this can compensate for larger values of "c" and higher thresholds (how can it do that?)
        - Importance of individual genes - how much do they influence final classification - how do gene levels correlate with prediction outcomes? 
        - Averaged importance predictiosn for top selected models 
        - Turns out that top-rated genes involved in cell proliferation control - their mutations associated often with cancer  

    - Pathway analytics
        - As noted above they identified the genes/proteins that were most influential for predicting anti-cancer therapeutics
        - They ran pathway analytics using gene-set enrichment (what is a pathway? what is pathway analytics?)
        - Among the top 25 impacted pathways were 
            - Cell cycle
            - DNA replication
            - Apoptosis
            - $p$-53 signalling
            - JAK-STAT signalling 
            - Mismatch repair
            - Also some cancer specific pathways 
        - These pathways are those that tend to be implicated in cancer development and progression
        - What do they mean by 'relative discriminating capacity' of a protein?
        - Many pathways are involved in cancer and most of the ones derived from the analysis have been suggested as targets for cancer prevention or interventions
        - An ideal agent should be able to disrupt multiple pro-tumorigenic (tumourigenic?) biochemical processes.
    - Drug repositioning
        - Most compounds used in cancer therapeutics had strong anti-cancer probability (according to the model?) 
        - As did some other compounds
        - One limitation of the approach is that they can identify which molecules interact but not how they interact (e.g. inhibit or stimulate).
    - Food molecules
        - Similar analysis as for drug molecules was done for food molecules
        - Food molecules in isolation might be less effective than in combination with other molecules
        - Anti-cancer properties will be determined by:
            1. Additive, antagonistic and synergistic actions of individual components
            2. How these simultaneously modulate different intracellular oncogenic pathways
    -          



    









