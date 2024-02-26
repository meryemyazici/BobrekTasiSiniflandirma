clc; clear all; close all;

% Veri dizinini tanımla
dosyaDizini = 'C:\Users\MERYEM\OneDrive\Belgeler\MATLAB\bobrekTasiProje\bobrekTasiVeri';
dosyalar = dir(fullfile(dosyaDizini, '*.*g'));

% Squeezenet ve DenseNet modellerini yükle
netA = squeezenet;
netB = densenet201;
layerA = "pool10"; % Squeezenet için özellik çıkarmak için kullanılacak katman
layerB = "conv5_block32_2_conv"; % Densenet201 için özellik çıkarmak için kullanılacak katman

inputSizeA = netA.Layers(1).InputSize; % Squeezenet için giriş boyutunu al
inputSizeB = netB.Layers(1).InputSize; % Densenet201 için giriş boyutunu al

% Parça boyutu
patchSize = 112; % Resimleri daha küçük parçalara ayırmak için kullanılacak parça boyutu (örneğin, 224x224 boyutundaki bir resmi yarıya bölmek)


% Özellikler ve etiketler için matrisler
X = []; % Özelliklerin saklanacağı boş matris
y = []; % Etiketlerin saklanacağı boş matris

% Veri kümesindeki her dosya için döngü
for i = 1:length(dosyalar)
    dosyaYolu = fullfile(dosyaDizini, dosyalar(i).name); % Dosyanın tam yolunu oluştur
    res = imread(dosyaYolu); % Görüntüyü oku
    
    % Gri tonlamalı ise RGB'ye dönüştür
    if size(res, 3) == 1
        res = cat(3, res, res, res); % Tek kanallı (gri tonlamalı) resmi üç kanallı (RGB) resme dönüştür
    end
    
    % Resmi her iki model için uygun boyuta yeniden boyutlandır
    resA = imresize(res, inputSizeA(1:2));
    resB = imresize(res, inputSizeB(1:2));

    % Her iki model için özellikleri çıkar
    featuresA = activations(netA, resA, layerA, 'OutputAs', 'rows'); % Squeezenet için uygun boyuta getir
    featuresB = activations(netB, resB, layerB, 'OutputAs', 'rows'); % Densenet201 için uygun boyuta getir
    
    % Özellikleri birleştir
    featuresCombinedFull = [featuresA, featuresB];

    featuresPatches = [];
    featuresCombinedPatches = [];
    for ii = 1:2
        for jj = 1:2
            % Resmin her bir parçasını (patch) ayıklayarak işle
            patch = res(1 + (ii-1)*patchSize:ii*patchSize, 1 + (jj-1)*patchSize:jj*patchSize, :);
            patchResizedA = imresize(patch, inputSizeA(1:2));
            patchResizedB = imresize(patch, inputSizeB(1:2));
            % Her bir model için bu parçalardan özellikleri çıkar
            featuresPatchA = activations(netA, patchResizedA, layerA, 'OutputAs', 'rows');
            featuresPatchB = activations(netB, patchResizedB, layerB, 'OutputAs', 'rows');
            % Bu özellikleri birleştir ve depola
            featuresCombinedPatches = [featuresCombinedPatches, featuresPatchA,featuresPatchB]; 
            %featuresCombinedPatches = [featuresPatches, featuresPatch]; % Özellikleri yatay olarak birleştir
        end
    end

    % Orijinal resim ve parça özelliklerini birleştir
    featuresCombined = [featuresCombinedFull, featuresCombinedPatches];
    X = [X; featuresCombined]; % Birleştirilmiş özellikleri X'e "- ekle
    y = [y; str2num(dosyalar(i).name(1))]; % Etiketi ekle
end

%Normalizasyon Bu satır, X matrisinin özelliklerini normalleştirir. 
% Her bir özelliği, o özelliğin minimum ve maksimum değerleri arasında ölçeklendirir. 
% eps terimi, sıfıra bölünmeyi önlemek için kullanılır.
Xx=(X-min(X))./(max(X)-min(X)+eps); 

% Chi-kare istatistiği kullanarak özellik seçimi yap
[idx,scores] = fscchi2(Xx,y);

% En önemli 2000 özelliği seç
 for i=1:2000
     son(:,i) = Xx(:,idx(i));
 end
 % Seçilen özelliklerin yanına etiketleri (y) ekleyerek matrisi tamamla
son(:, i+1) = y;

