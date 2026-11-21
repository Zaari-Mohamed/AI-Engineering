recherche semantique avec sentence transformers mais sans base vectorielle (vecteurs stockés en local)


1- on doit preciser le model de embedding  qui est le sentence transformers
2- puis on doit faire encoder tous les elemenst de docs (les phrases) pour les transforemer en vecterrs mais stocker dan vects docs en local a chaque fois qu'on run le code il se genere (pas une bonne pratique )
3- apres on doit faire une focntion de recherche semantique qui va prendre la requete de user puis calcule le cos entre les deu  verceturs et aussi fait le score pour chacundans docs .
4- affiches les trois les pus proche de chque requete (boucle for )