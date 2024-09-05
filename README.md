# Lipreading-Using-Deep-Neural-Networks-Python

ABSTRACT

The project endeavors to develop a Lip-reading system utilizing deep learning techniques to mitigate communication barriers faced by individuals with hearing impairments. Central to the initiative is the adaptation of the LipNet architecture, preprocessing video input into sequential data for analysis. This architecture integrates Conv3D layers followed by bi-directional GRU layers with ReLU activation functions, enabling the model to adeptly capture spatial and temporal information from lip movements. A pivotal component is the implementation of the Connectionist Temporal Classification (CTC) loss function during model training. This function ensures efficient training by enabling the model to learn from sequences without necessitating alignment between input and output data.

The project's core component is a modified version of the powerful deep learning model LipNet, which was created especially for lip reading applications. This design includes bi-directional LSTM (Long Short-Term Memory) layers that record the temporal relationships inside the lip motions, as well as Conv3D layers that extract spatial-temporal information from video frame sequences. The model includes non-linearity by using ReLU activation functions, which allows it to learn intricate correlations between linguistic units and visual inputs.

DEMO

https://github.com/user-attachments/assets/490df2be-8a11-47cc-af57-7d092208f082

