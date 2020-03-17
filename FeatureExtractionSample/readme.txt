Use the below given syntax to extract features

./ComputeFeatures [configfile] [wavefile] [Feature-name] [filename-name] [Thresold-for-vad] [channel-to-use]


configfile - contains various confirguration for extracting features. Example configuration for extracting mfcc is given in "mfcc.config".

Feature-name - use "frameCepstrum+frameDeltaCepstrum" for mfc coefficients and delta coefficients 

Thresold-for-vad - Thresold used to comptue VAD.

channel-to-use -  Channel to use in case of multichannel data. For timit dataset use "A" as channel.


Example command to extract mfcc features for sa1.wav

./ComputeFeatures mfcc.config  sa1.wav frameCepstrum+frameDeltaCepstrum sa1.mfcc 0.06 A
