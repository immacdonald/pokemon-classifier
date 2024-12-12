# Pokemon Classifier Dataset & ML Models

This repository contains code for a machine learning research project involving first gathering a novel dataset of Pokemon images, which is available in the [dataset/](/dataset/) folder, and then using transfer learning to train a machine learning model to predict key attributes (types) of a Pokemon based on its design. The models achieved accuracy exceeding 85% for both single and multi-type classification problems.

## Dataset

The dataset created for this project contains ~36,000 Pokemon images collected by [scraping](/scraping/) the [Bulbagarden Archives](https://archives.bulbagarden.net/wiki/Main) as well as [PKMNCards](https://pkmncards.com/). The images are categorized based on individual Pokemon species, further separated by alternate forms such as regional and mega Pokemon. In total there are 1156 distinct Pokemon forms included.

### The Pokedex

Included in this project is a "Pokedex" which contains information on the 1156 Pokemon forms. This includes their Pokedex number, an id (derived from Pokedex number and the form, if applicable), primary and secondary types, and the Pokemon's generation. This data is stored in a JSON file in [pokedex/pokemon_list.json](/pokedex/pokemon_list.json) and is accurate for 2024 (contains up to Pokemon #1025).

### Unpacking the Image Data

To use the dataset first unzip the [data.zip](/dataset/data.zip) file. 

Then, use the [unpack_dataset.py](/dataset/unpack_dataset.py) script to automatically resort the data into either single-type class folders or one-hot encoded multi-class images. This can be done with the following command:

```
python3 -m dataset.unpack_dataset
```

### Notes on the Dataset

The dataset was filtered and sorted from a starting batch of over a quarter million Pokemon images. Cleaning the data involved both automated and manual review:
1. Run scrapers for Bulbapedia and Pokemon Cards
2. Filter Bulbapedia 
   - Remove images that are considered invalid (non-RGB(A) image mode, too much blank space, too small)
   - Scripted pass at moving images to form folders based on name (such as images with "Alola" for an Alolan form)
   - Manually review images to remove any that do not primarily focus on the Pokemon and to move any remaining alternate forms
3. Filter Pokemon Cards
   - Run scrapers for the exlusion lists (shiny, alternate forms, etc.)
   - Move or delete images based on exclusion lists
   - Manual review for other forms, validate that no new card variants need to be excluded
   - Resize script

## ML Models & Training

Training for this project was done using [PyTorch](https://pytorch.org) with transfer learning and fine-tuning of [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

### Dataset Split Methods

Two different methods were used when splitting the data into the training, validation, and testing subsets. First was a fully random split which distributed images between the subsets using the desired ratio (which for this project was 70% / 15% / 15%). This created the most even distribution with each subset containing images of virtually all Pokemon and forms. Another method was used as well - a stratified split which constrains all images of any given Pokemon to only be in one of the subsets. This was done to determine how much the models were learning about the visual elements of each type vs. how much they were able to predict a Pokemon from similar images of that Pokemon.

### Results

A correlation between Poke ́mon designs and types is shown to exist and is viable for classification tasks. Given that randomly guessing a Pokemon’s type would yield only 5.5% accuracy on average, and even less when attempting to guess monotype and dual-type Pokemon, all models displayed significantly stronger results.

#### Single-Type Classification

Single-type classification using only the primary type of each Pokemon achieved an accuracy of 90.4% using the random split and 52.3% for the stratified split.


#### Multi-Type Classification

Multi-type classification taking in account all types of each Pokemon achieved an accuracy of 87.3% using the random split and 41.1% for the stratified split.