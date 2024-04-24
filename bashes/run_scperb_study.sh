# python3 -W ignore scperb.py --model_name CD4T --exclude_celltype CD4T --data pbmc
# python3 -W ignore scperb.py --model_name CD14+Mono --exclude_celltype CD14+Mono --data pbmc
# python3 -W ignore scperb.py --model_name B --exclude_celltype B --data pbmc
# python3 -W ignore scperb.py --model_name CD8T --exclude_celltype CD8T --data pbmc
# python3 -W ignore scperb.py --model_name NK --exclude_celltype NK --data pbmc
# python3 -W ignore scperb.py --model_name FCGR3A+Mono --exclude_celltype FCGR3A+Mono --data pbmc
# python3 -W ignore scperb.py --model_name Dendritic --exclude_celltype Dendritic --data pbmc

# python3 -W ignore scperb.py --model_name Endocrine --exclude_celltype Endocrine --data hpoly
# python3 -W ignore scperb.py --model_name Enterocyte --exclude_celltype Enterocyte --data hpoly
# python3 -W ignore scperb.py --model_name Enterocyte.Progenitor --exclude_celltype Enterocyte.Progenitor --data hpoly
# python3 -W ignore scperb.py --model_name Goblet --exclude_celltype Goblet --data hpoly
# python3 -W ignore scperb.py --model_name Stem --exclude_celltype Stem --data hpoly
# python3 -W ignore scperb.py --model_name TA --exclude_celltype TA --data hpoly
# python3 -W ignore scperb.py --model_name TA.Early --exclude_celltype TA.Early --data hpoly
# python3 -W ignore scperb.py --model_name Tuft --exclude_celltype Tuft --data hpoly

python3 -W ignore scperb.py --model_name CD4T --exclude_celltype CD4T --data study --supervise True
python3 -W ignore scperb.py --model_name CD14+Mono --exclude_celltype CD14+Mono --data study --supervise True
python3 -W ignore scperb.py --model_name B --exclude_celltype B --data study --supervise True
python3 -W ignore scperb.py --model_name CD8T --exclude_celltype CD8T --data study --supervise True
python3 -W ignore scperb.py --model_name NK --exclude_celltype NK --data study --supervise True
python3 -W ignore scperb.py --model_name FCGR3A+Mono --exclude_celltype FCGR3A+Mono --data study --supervise True
python3 -W ignore scperb.py --model_name Dendritic --exclude_celltype Dendritic --data study --supervise True