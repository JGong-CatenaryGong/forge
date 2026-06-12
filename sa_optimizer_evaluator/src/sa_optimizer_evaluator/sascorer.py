"""SA Score wrapper — delegates to RDKit Contrib sascorer."""

from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer


def calc_sa_score(smiles: str) -> float:
    """SA Score: 1(最简单) ~ 10(最难), Ertl & Schuffenhauer 2009."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    return round(sascorer.calculateScore(mol), 4)
