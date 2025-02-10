# The Contrast-proof

This is a proposal for an algorithm to evaluate the contrast of an image before it is printed, based on the tonal curve of a paper.

To do this, we must first print a densitometric scale on a printing paper and read it with a spectrophotometer. This curve, expressed in 8-bit count values ​​(255-0), is used to compare the tonal range of the image with the tonal response of the paper.

![stepwedge](https://github.com/jpereiranet/contrast-proof/blob/e7f67fde0c59c2ff027d654746605bd8d3e303cc/stepwedge/stepwedge.jpg)

Clipping areas or areas outside the tonal range of the image, whether in shadows or highlights, will be shown as a gray area.

The colors show the degree of contrast of the image, where blue is less contrast and yellow is more contrast.

Hahnemuhle Agave
![paper1](https://github.com/jpereiranet/contrast-proof/blob/74de950592dad11b99f8e19c77d19015bfa95896/output/proof_Agave.png )

Hahnemuhle Photo_Rag_Metallic
![paper2](https://github.com/jpereiranet/contrast-proof/blob/7917179c17c57cedbbe12706a7adcdc53849cc6c/output/proof_Photo_Rag_Metallic.png)

Hahnemuhle Baryta_FB
![paper2](https://github.com/jpereiranet/contrast-proof/blob/7917179c17c57cedbbe12706a7adcdc53849cc6c/output/proof_Baryta_FB.png)




# Usage

The algorithm uses a curve with this format, it can have different number of inputs depending on the size of the grayscale

"202,196,188,180,173,166,160,154,146,140,136,129,123,118,112,108,104,99,95,90,86,83,79,76,73,70,67,65,63,60,59,57,56,54,53,51"

You also need the path to an image file, in the test-imgs folder there is an example image

The papers.csv file has a list of example curves for various Hahnemuhle printing papers.

The file process.py only processes the rows of the CSV file as an example.

In the output folder there is an example of the output for those papers and image.

