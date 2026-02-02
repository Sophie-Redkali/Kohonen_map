% ===================================
% Analyse ROI - Identification des zones d'intérêt
% (Forêts, eau, végétation, sol nu, etc.)
% ===================================

close all;
clear;

% ===================================
% Chargement des données affichage image étudié
% ===================================

fprintf('Chargement des données...\n');
dossier = input('Entrez le chemin du dossier de résultats : ', 's');

load(fullfile(dossier, 'resultats_analyse_roi.mat'));

fprintf('Données chargées avec succès !\n');
fprintf('  Taille image : %d x %d\n', n, p);

Y_rgb = cat(3, Y(:,:,3), Y(:,:,2), Y(:,:,1)); % R,V,B
Y_rgb_norm = double(Y_rgb) / (2^16 - 1);

figure('Position', [100, 100, 1200, 500]);
imshow(Y_rgb_norm);
title('Image originale RGB');

% ============================================
% Calcul des indices pixel par pixel pour un placement sur l'image
% ============================================

fprintf('\nCalcul des indices spectraux par pixel...\n');

% (IR1 - ROUGE) / (IR1 + ROUGE)
NDVI_map = (Y(:,:,4) - Y(:,:,3)) ./ (Y(:,:,4) + Y(:,:,3) + eps);
% (VERT - IR2) / (VERT + IR2)
MNDWI_map = (Y(:,:,2) - Y(:,:,5)) ./ (Y(:,:,2) + Y(:,:,5) + eps);
% ((IR1 - ROUGE)/(IR1 + ROUGE + 0.5)) * 1.5
SAVI_map = ((Y(:,:,4) - Y(:,:,3)) ./ (Y(:,:,4) + Y(:,:,3) + 0.5)) * 1.5;
% (IR2 - IR1) / (IR2 + IR1)
NDBI_map = (Y(:,:,5) - Y(:,:,4)) ./ (Y(:,:,5) + Y(:,:,4) + eps);
% IR1 / ROUGE
NIR_Red_ratio = Y(:,:,4) ./ (Y(:,:,3) + eps);


figure('Position', [100,100,1400,400]);
subplot(2,3,1);
imagesc(NDVI_map);
colorbar; axis square;
title('NDVI');

subplot(2,3,2);
imagesc(MNDWI_map);
colorbar;
axis square;
title('MNDWI');

subplot(2,3,3);
imagesc(NIR_Red_ratio);
colorbar;
axis square;
title('NIR/Red Ratio');

subplot(2,3,4);
imagesc(SAVI_map);
colorbar;
axis square;
title('SAVI');

subplot(2,3,5);
imagesc(NDBI_map);
colorbar;
axis square;
title('NDBI');

% ===================================
% Classification des zones (la délimitation des zones
% manuellement ne fonctionne pas correctement avec Octave)
% ===================================

fprintf('\nClassification pixel par pixel...\n');

image_classifiee = zeros(n,p);

for i = 1:n
    for j = 1:p
        NDVI = NDVI_map(i,j);
        MNDWI = MNDWI_map(i,j);
        SAVI = SAVI_map(i,j);
        NDBI = NDBI_map(i,j);

        if MNDWI > 0.0
            image_classifiee(i,j) = 1; % Eau
        elseif NDBI > 0.3
            image_classifiee(i,j) = 4; % Sol nu / urbain
        elseif NDVI > 0.5
            image_classifiee(i,j) = 2; % Forêt
        elseif SAVI > 0.2
            image_classifiee(i,j) = 3; % Végétation
        else
            image_classifiee(i,j) = 4; % Sol nu par défaut
        end
    end
end

% ===================================
% Les classes : pourcentage & affichage
% ===================================

nb_pixels_eau = sum(image_classifiee(:) == 1);
nb_pixels_foret = sum(image_classifiee(:) == 2);
nb_pixels_veg = sum(image_classifiee(:) == 3);
nb_pixels_sol = sum(image_classifiee(:) == 4);
total_pixels = n * p;

fprintf('\nStatistiques par classe :\n');
fprintf('  Eau           : %d pixels (%.1f%%)\n', nb_pixels_eau, 100*nb_pixels_eau/total_pixels);
fprintf('  Forêt         : %d pixels (%.1f%%)\n', nb_pixels_foret, 100*nb_pixels_foret/total_pixels);
fprintf('  Végétation    : %d pixels (%.1f%%)\n', nb_pixels_veg, 100*nb_pixels_veg/total_pixels);
fprintf('  Sol nu/Urbain : %d pixels (%.1f%%)\n', nb_pixels_sol, 100*nb_pixels_sol/total_pixels);

masque_eau   = (image_classifiee == 1);
masque_foret = (image_classifiee == 2);
masque_veg   = (image_classifiee == 3);
masque_sol   = (image_classifiee == 4);

% Affichage des diiférents rendus et de l'image d'origine
figure('Position', [100, 100, 1400, 900]);
subplot(2,3,1);
imshow(Y_rgb_norm);
title('Image originale');

subplot(2,3,2);
imshow(masque_eau);
title(sprintf('Eau (%d pixels)', nb_pixels_eau));

subplot(2,3,3);
imshow(masque_foret);
title(sprintf('Forêt (%d pixels)', nb_pixels_foret));

subplot(2,3,4);
imshow(masque_veg);
title(sprintf('Végétation (%d pixels)', nb_pixels_veg));

subplot(2,3,5);
imshow(masque_sol);
title(sprintf('Sol nu (%d pixels)', nb_pixels_sol));

% Superposition colorée
subplot(2,3,6);
RGB_classif = zeros(n,p,3);
RGB_classif(:,:,3) = double(masque_eau);
RGB_classif(:,:,2) = double(masque_foret)*0.8;
RGB_classif(:,:,1) = double(masque_veg)*0.5;
RGB_classif(:,:,2) = RGB_classif(:,:,2) + double(masque_veg);
RGB_classif(:,:,1) = RGB_classif(:,:,1) + double(masque_sol)*0.8;
RGB_classif(:,:,2) = RGB_classif(:,:,2) + double(masque_sol)*0.6;
RGB_classif(:,:,3) = RGB_classif(:,:,3) + double(masque_sol)*0.4;
imshow(RGB_classif);
title('Superposition colorée');

% =============================
% Sauvegardes
% =============================

resultats_classification = struct();
resultats_classification.image_classifiee = image_classifiee;
resultats_classification.masque_eau = masque_eau;
resultats_classification.masque_foret = masque_foret;
resultats_classification.masque_veg = masque_veg;
resultats_classification.masque_sol = masque_sol;
resultats_classification.NDVI_map = NDVI_map;
resultats_classification.MNDWI_map = MNDWI_map;
resultats_classification.statistiques.nb_pixels_eau = nb_pixels_eau;
resultats_classification.statistiques.nb_pixels_foret = nb_pixels_foret;
resultats_classification.statistiques.nb_pixels_veg = nb_pixels_veg;
resultats_classification.statistiques.nb_pixels_sol = nb_pixels_sol;

save(fullfile(dossier, 'classification_resultats.mat'), 'resultats_classification');

imwrite(uint8(image_classifiee*60), fullfile(dossier,'image_classifiee.png'));
imwrite(masque_eau, fullfile(dossier,'masque_eau.png'));
imwrite(masque_foret, fullfile(dossier,'masque_foret.png'));
imwrite(masque_veg, fullfile(dossier,'masque_vegetation.png'));
imwrite(masque_sol, fullfile(dossier,'masque_sol_nu.png'));

fprintf('\nRésultats sauvegardés dans : %s\n', dossier);
fprintf('\n===\nANALYSE TERMINÉE\n===\n');

