# Contactless-Heartrate-detectio
An approach to mesure heart rate from video feed using euler magnification, PPG (Photoplethysmorgan) solution based on a color magnification algorithm
which makes it possible to see the color of your face change as blood rushes in and out of your head.

Approach:
1. Gaussian Pyramid:
   This was done to extract features and remove noise
2. Time Domain filtering:
   This time domain band filtering is done to obtain only the frequency bands of interest.
3. Amplify the filtering result:
   Results of will be amplified, and then approximated.


References:
https://medium.com/intel-software-innovators/heartrate-detection-using-camera-d34b3289e272
https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://github.com/habom2310/Heart-rate-measurement-using-camera
http://people.csail.mit.edu/mrub/papers/vidmag.pdf
   
