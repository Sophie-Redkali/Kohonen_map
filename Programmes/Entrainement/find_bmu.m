function [bmu_lig, bmu_col] = find_bmu(som_map, pixel)
  [nb_ligne, nb_colonne, dimension] = size(som_map);
  min_dist = inf;
  bmu_lig = 1;
  bmu_col = 1;
  for l = 1:nb_ligne
    for c = 1:nb_colonne
      neurone = squeeze(som_map(l,c,:))';
      dist = sum((pixel-neurone).^2);
      if dist < min_dist || (dist == min_dist && rand() < 0.5)
        min_dist = dist;
        bmu_lig = l;
        bmu_col = c;
      end
    end
  end
end

