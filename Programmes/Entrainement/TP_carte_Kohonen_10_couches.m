% ===================================
% Carte de Kohonen appliquées aux SIG
% ===================================
close all;
clear;

% ===================================
% Chargement de la portion d'image à étudier
% ===================================
% Coordonnées utilisé sur les images Landsat
x_debut = 4301;
x_fin = 4350;
y_debut = 3651;
y_fin = 3700;

% Couche bleu
B = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B2.TIF');
B = B(x_debut:x_fin, y_debut:y_fin);
B = (B-7500)*10.1;

% Couche vert
V = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B3.TIF');
V = V(x_debut:x_fin, y_debut:y_fin);
V = (V-7500)*10.1;

% Couche rouge
R = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B4.TIF');
R = R(x_debut:x_fin, y_debut:y_fin);
R = (R-7500)*10.1;

% Couche infra-rouge 851nm-879nm
I1 = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B5.TIF');
I1 = I1(x_debut:x_fin, y_debut:y_fin);
I1 = I1*2.62;

% Couche infra-rouge 1566nm-1651nm
I2 = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B6.TIF');
I2 = I2(x_debut:x_fin, y_debut:y_fin);
I2 = I2*2.62;

% Couche infra-rouge 2107nm-2294nm
I3 = imread('LC09_L2SP_199026_20230909_20230911_02_T1_SR_B7.TIF');
I3 = I3(x_debut:x_fin, y_debut:y_fin);
I3 = I3*2.62;

n = size(B,1);
p = size(B,2);
dim = 10;
Y = zeros(n, p, dim);
Y(:,:,1) = B;
Y(:,:,2) = V;
Y(:,:,3) = R;
Y(:,:,4) = I1;
Y(:,:,5) = I2;
Y(:,:,6) = I3;

for i = 1:n
  for j = 1:p
    % Création des diférentes couches calculés pour analyse
    VERT = double(Y(i,j,2));
    ROUGE = double(Y(i,j,3));
    IR1 = double(Y(i,j,4));
    IR2 = double(Y(i,j,5));
    IR3 = double(Y(i,j,6));
    % Calcul des différentes couches
    % NDVI Végétation
    NDVI = (IR1 - ROUGE) / (IR1 + ROUGE);
    % MNDWI Eau ou zone humide : peut être utile pour comparer une zone avant / après une crue
    MNDWI = (VERT - IR2) / (VERT + IR2);
    % NDBI Bâti ou sol nu
    NDBI = (IR2 - IR1) / (IR2 + IR1);
    % SAVI végétation sol clair
    SAVI = ((IR1 - ROUGE)/(IR1 + ROUGE + 0.5)) * 1.5;
    % Mise à jour de Y
    Y(i,j,7) = NDVI;
    Y(i,j,8) = SAVI;
    Y(i,j,9) = MNDWI;
    Y(i,j,10) = NDBI;
  end
end

for i = 1:dim
  Y(:,:,i) = max(0, min(Y(:,:,i), 2^16 -1));
end

% Afficher l'image en couleur vraie
Y_vrai = cat(3, R, V, B);
figure;
imagesc(Y_vrai, [0, 2^16 - 1]);
axis square;
title('Image reconstituée');

% ===================================
% Matrice X contenant toutes les couches et paramétrage du programme
% ===================================
% Construction de la matrice X
disp("Utilisation de reshape pour construire X")
X = reshape(Y, [n*p, dim]);

% Paramètres
nx = input('Entrez le nombre de lignes de la carte SOM : ');
ny = nx;
stop = input('Entrez le nombre itération souhaité : ');
epsilon = input('Entrer la valeur de départ du epsilon : ');
rmax = input('Rayon max de départ autour du BMU : ');

% ===================================
% Entrainement
% ===================================
% Appel de la fonction kohonen_som
[Map, N] = kohonen_som(X, nx, ny, epsilon, rmax, stop);

fprintf('\n--- Construction des labels de clusters (neurones SOM) ---\n');

% Met chaque neurone sur une ligne
d = size(Map, 3);
W = reshape(Map, nx * ny, d);

% ===================================
% Classification des pixels par k-means
% ===================================
% Choix de créer 6 classes
nb_clusters = 6;

% k-means sur les neurones SOM : impossible d'installer
% le package statistique dans Octave.
max_iter = 50;

% initialisation aléatoire des centres
idx0 = randperm(size(W,1), nb_clusters);
centres = W(idx0, :);

clusters = zeros(size(W,1),1);

for iter = 1:max_iter
    old_clusters = clusters;

    % Assignation
    for i = 1:size(W,1)
        d = sum((centres - W(i,:)).^2, 2);
        [valeur_min, idx_min] = min(d);
        clusters(i) = idx_min;
    end

    % Mise à jour centres
    for k = 1:nb_clusters
        membres = W(clusters == k, :);
        % ~ispemty = non vide, le tilde est la négation logique
        if ~isempty(membres)
            centres(k,:) = mean(membres, 1);
        end
    end

    % Arrêt si convergence
    if isequal(old_clusters, clusters)
        fprintf('Convergence à l''itération %d\n', iter);
        break;
    end
end

% Affichage de la carte SOM RGB après entraînement
som_rvb = Map(:,:,1:3);
som_rvb_affichage = double(som_rvb) / (2^16 - 1);
figure;
imshow(som_rvb_affichage);
title('Carte SOM après entrainement - RGB');

% ===================================
% Création de l'image colorisée par classe
% ===================================
fprintf('\nCréation de la carte colorisée...\n');

% Palette pour la coloration des classes
%  Bleu (eau)
%  Vert (végétation dense)
%  Beige (sol nu)
%  Rouge (urbain / bâti)
%  Orange (transition)
%  Gris (autres)
colors = [
    0,   80,   200;
    0,   160, 60;
    230, 210, 120;
    200, 70, 70;
    255, 170, 50;
    180, 180, 180;
] / 255;

seg_color = zeros(n, p, 3);

for i = 1:n
    for j = 1:p
        k = labels_image(i,j);
        if k > 0 && k <= size(colors,1)
            seg_color(i,j,:) = colors(k,:);
        end
    end
end

figure;
imshow(seg_color);
title('Segmentation colorisée');

% ===================================
% Extraction des données pour analyse ROI
% ===================================

fprintf('\nExtraction des données pour analyse ROI...\n');
% Extraire les 10 couches de Map
carte_SOM_B  = Map(:,:,1);
carte_SOM_V  = Map(:,:,2);
carte_SOM_R  = Map(:,:,3);
carte_SOM_I1 = Map(:,:,4);
carte_SOM_I2 = Map(:,:,5);
carte_SOM_I3 = Map(:,:,6);
carte_SOM_NDVI = Map(:,:,7);
carte_SOM_SAVI = Map(:,:,8);
carte_SOM_MNDWI = Map(:,:,9);
carte_SOM_NDBI = Map (:,:,10);
carte_SOM_RGB = Map(:,:,1:3);

% Visualiser les 6 cartes SOM
figure('Position', [100, 100, 1400, 900]);

subplot(2, 5, 1);
imagesc(carte_SOM_B);
colorbar;
axis square;
title('SOM - Bleu');

subplot(2, 5, 2);
imagesc(carte_SOM_V);
colorbar;
axis square;
title('SOM - Vert');

subplot(2, 5, 3);
imagesc(carte_SOM_R);
colorbar;
axis square;
title('SOM - Rouge');

subplot(2, 5, 4);
imagesc(carte_SOM_I1);
colorbar;
axis square;
title('SOM - IR1');

subplot(2, 5, 5);
imagesc(carte_SOM_I2);
colorbar;
axis square;
title('SOM - IR2');

subplot(2, 5, 6);
imagesc(carte_SOM_I3);
colorbar;
axis square;
title('SOM - IR3');

subplot(2, 5, 7);
imagesc(carte_SOM_NDVI);
colorbar;
axis square;
title('SOM - NDVI');

subplot(2, 5, 8);
imagesc(carte_SOM_SAVI);
colorbar;
axis square;
title('SOM - SAVI');

subplot(2, 5, 9);
imagesc(carte_SOM_MNDWI);
colorbar;
axis square;
title('SOM - MNDWI');

subplot(2, 5, 10);
imagesc(carte_SOM_NDBI);
colorbar;
axis square;
title('SOM - NDBI');

% ===================================
% Sauvegarde des résultats
% ===================================

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
dossier_resultats = ['resultats_', timestamp];
mkdir(dossier_resultats);

fprintf('\nSauvegarde des résultats dans : %s\n', dossier_resultats);

% Images
imwrite(uint16(Y_vrai), fullfile(dossier_resultats, 'image_originale_RGB.tif'));
imwrite(som_rvb_affichage, fullfile(dossier_resultats, 'som_RGB.png'));
imwrite(uint16(labels_image), fullfile(dossier_resultats, 'image_segmentee.tif'));
imwrite(seg_color, fullfile(dossier_resultats, 'segmentation_colorisee.png'));

% Paramètres
parametres = struct();
parametres.nx = nx;
parametres.ny = ny;
parametres.stop = stop;
parametres.epsilon = epsilon;
parametres.rmax = rmax;
parametres.taille_image = [n, p];
parametres.nb_pixels = n*p;
parametres.timestamp = timestamp;

save(fullfile(dossier_resultats, 'parametres.mat'), 'parametres');
% IMPORTANT : Sauvegarder Y pour l'analyse ultérieure
save(fullfile(dossier_resultats, 'Y_original.mat'), 'Y');

% Données pour analyse ROI
save(fullfile(dossier_resultats, 'resultats_analyse_roi.mat'), ...
     'carte_SOM_B', 'carte_SOM_V', 'carte_SOM_R', ...
     'carte_SOM_I1', 'carte_SOM_I2', 'carte_SOM_I3', ...
     'carte_SOM_NDVI', 'carte_SOM_SAVI', 'carte_SOM_MNDWI', 'carte_SOM_NDBI', ...
     'carte_SOM_RGB', 'labels_image', 'N', 'Map', ...
     'parametres', 'nx', 'ny', 'n', 'p', 'Y');

% Matrice N
figure;
imagesc(N);
colorbar;
axis square;
title('Nombre de pixels par nœud');
print(fullfile(dossier_resultats, 'matrice_N.png'), '-dpng');
dlmwrite(fullfile(dossier_resultats, 'matrice_N.txt'), N, '\t');

% Rapport
fid = fopen(fullfile(dossier_resultats, 'rapport.txt'), 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'Rapport - Carte de Kohonen\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date : %s\n\n', datestr(now));
fprintf(fid, 'Paramètres :\n');
fprintf(fid, '  - Taille carte SOM : %d x %d\n', nx, ny);
fprintf(fid, '  - Nombre de neurones : %d\n', nx*ny);
fprintf(fid, '  - Itérations : %d\n', stop);
fprintf(fid, '  - Epsilon : %.3f\n', epsilon);
fprintf(fid, '  - Rmax : %.2f\n', rmax);
fprintf(fid, '  - Taille image : %d x %d\n', n, p);
fprintf(fid, '  - Nombre de pixels : %d\n', n*p);
fprintf(fid, '  - Canaux : 10\n\n');
fprintf(fid, 'Matrice N :\n');
fprintf(fid, '  - Min pixels/nœud : %d\n', min(N(:)));
fprintf(fid, '  - Max pixels/nœud : %d\n', max(N(:)));
fprintf(fid, '  - Moyenne : %.2f\n', mean(N(:)));
fprintf(fid, '  - Écart-type : %.2f\n', std(N(:)));
fprintf(fid, '  - Nœuds vides : %d/%d (%.1f%%)\n', ...
        sum(N(:)==0), nx*ny, 100*sum(N(:)==0)/(nx*ny));
fprintf(fid, '  - Total classifiés : %d\n', sum(N(:)));
fclose(fid);

fprintf('\n========================================\n');
fprintf('Sauvegarde terminée\n');
fprintf('========================================\n');
fprintf('Dossier : %s\n', dossier_resultats);
fprintf('========================================\n');


