% Création de la fonction pour l'entrainement de la carte auto-adaptative
function [Map,N] = kohonen_som(X, nx, ny, epsilon, rmax, stop)

  % Récupérer le min et le max de chaque colonne de la matrice X
  val_min = min(X, [], 1);
  val_max = max(X, [], 1);

  disp('Min de chaque colonne : ');
  disp(val_min);
  disp('Max de chaque colonne : ');
  disp(val_max);

  [nb_pixel, nb_couche] = size(X);

  % Initialisation de la carte SOM par échantillonage
  rand('seed', sum(100*clock));
  nb_neurones = nx * ny;
  indices_random = randperm(nb_pixel, nb_neurones);
  poids_init = X(indices_random, :);
  Map = reshape(poids_init, nx, ny, nb_couche);


  % Affichage de la carte SOM avant entrainement
  som_rvb = Map(:,:,1:3);
  som_rvb_affichage = double(som_rvb) / (2^16 - 1);
  figure;
  imshow(som_rvb_affichage);
  title('Carte SOM avant entrainement');

  % =======================
  %   ENTRAINEMENT
  % =======================
  for iter = 1:stop

    epsilon_iter = max(0.05, epsilon * (1 - iter/stop));
    rmax_iter = max(1, rmax * exp((-3*iter/stop)));

    % Affichage compteur itération
    if mod(iter, 2) == 0
      fprintf('Itération %d/%d (%.1f%%) - epsilon=%.4f, rmax=%.2f\n', iter, stop, (iter/stop)*100, epsilon_iter, rmax_iter);
    end

    % Choix aléatoire des pixels
    idx_aleatoire = randperm(nb_pixel);

    for idx = 1:length(idx_aleatoire)
        pixel = X(idx_aleatoire(idx),:);

        % Appel de la fonction find_bmu
        [bmu_lig, bmu_col] = find_bmu(Map, pixel);

        % Avancement des itérations
        if mod(idx, 250) == 0
            fprintf('Pixel %d/%d (%.1f%%)\n', idx, length(idx_aleatoire), (idx/length(idx_aleatoire))*100);
        end

        % Mise à jour
        for l = 1:nx
            for c = 1:ny
                dist_bmu = sqrt((l-bmu_lig)^2 + (c-bmu_col)^2);
                if dist_bmu <= rmax_iter
                    h = exp(-dist_bmu^2 / (2*rmax_iter^2));
                    w = squeeze(Map(l,c,:))';                 % 1xdim
                    w = w + epsilon_iter * h * (pixel - w);   % mise à jour
                    Map(l,c,:) = reshape(w, 1, 1, nb_couche); % remettre dans Map
                end
            end
        end
    end

    if mod(iter, 50) == 0
        fprintf('Itération %d/%d - epsilon=%.3f, rmax=%.2f\n', iter, stop, epsilon_iter, rmax_iter);
    end

    % ===============================
    %  VISUALISATIONS ET SAUVEGARDE EN COURS
    % ===============================
    if mod(iter, 10) == 0      % tous les 10 itérations
        % Construction d'une version RGB de la SOM
        som_rgb = Map(:,:,1:3);
        som_rgb_aff = double(som_rgb) / (2^16 - 1);

        % Affichage
        figure(100);
        clf;
        imshow(som_rgb_aff);
        title(sprintf('SOM - après %d / %d itérations', iter, stop));
        drawnow();   % force l’affichage immédiat

        % --- SAUVEGARDE ---
        nom_dossier_temp = 'SOM_intermediaire';
        if ~exist(nom_dossier_temp, 'dir')
            mkdir(nom_dossier_temp);
        end

        % Sauvegarde de la carte SOM RGB
        imwrite(som_rgb_aff, fullfile(nom_dossier_temp, sprintf('SOM_RGB_iter_%03d.png', iter)));

        % Sauvegarde de la matrice N jusqu’ici
        N_temp = zeros(nx, ny);
        for idx_pixel = 1:nb_pixel
            pixel_temp = X(idx_pixel,:);
            [bmu_lig, bmu_col] = find_bmu(Map, pixel_temp);
            N_temp(bmu_lig, bmu_col) = N_temp(bmu_lig, bmu_col) + 1;
        end
        save(fullfile(nom_dossier_temp, sprintf('N_iter_%03d.mat', iter)), 'N_temp');
    end
  end

  % ================================
  %  MATRICE N FINALE
  % ================================
  N = zeros(nx, ny);
  for idx = 1:nb_pixel
    pixel = X(idx,:);
    [bmu_lig, bmu_col] = find_bmu(Map, pixel);
    N(bmu_lig, bmu_col) = N(bmu_lig, bmu_col) + 1;
  end

end

