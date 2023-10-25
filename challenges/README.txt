Le challenge 2 concerne l'inférence du taux de mutation quand les autres paramètres sont connus, comme vous l'aviez déjà fait "à la main" sur un cas particulier (challenge 1), mais avec 100 jeux de données différents, et il s'agit donc de faire appel à votre fonction d'inférence complètement automatisée.

Chaque fichier de ce challenge contient donc 100 lignes (une par jeu de données): challenge2.data contient les 100 jeux de données (96 nombres de mutants -- car 96 réplicats -- par jeu de données), challenge2.d contient les 100 valeurs du taux de mortalité, challenge2.f les valeurs de l'effet sur la fitness de la mutation, challenge2.p les valeurs du taux d'echantillonage, challenge2.N les valeurs de la taille de population finale. Il faut bien garder la correspondance entre les lignes : la ligne 1 correspond au 1er jeu de données, etc.

Le challenge 3 est organisé de la même façon, mais cette fois le taux de mutation et l'effet de la mutation sur la fitness sont tous les deux inconnus.

Sortie attendu : des fichiers challenge2.mrate, challenge3.mrate et challenge3.f, suivant le même format (une valeur par ligne en gardant bien la correspondance).
Si vous n'avez pas encore de code terminé pour challenge3, faites seulement challenge2. Si votre code est trop lent pour traiter les 100 jeux de données, faites en autant que possible.
