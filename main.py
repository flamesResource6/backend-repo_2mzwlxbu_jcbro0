import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import requests

# Database helpers (lazy / safe import)
try:
    from database import db, create_document, get_documents  # type: ignore
except Exception:
    db = None  # type: ignore

    def create_document(*args, **kwargs):  # type: ignore
        return None

    def get_documents(*args, **kwargs):  # type: ignore
        return []

app = FastAPI(title="Food Vision + Nutrition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Utility helpers
# -----------------------------

class MacroEstimation(BaseModel):
    kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    uncertainty: float = Field(0.35, ge=0.0, le=1.0)
    notes: Optional[str] = None

class Plausibility(BaseModel):
    per_100g_kcal: Optional[float] = None
    within_bounds: bool = True
    message: Optional[str] = None

class A1Response(BaseModel):
    filename: Optional[str]
    size_bytes: Optional[int]
    items: List[Dict[str, Any]] = Field(default_factory=list)
    portion_grams: Optional[float] = None
    estimation: MacroEstimation
    plausibility: Plausibility
    created_at: datetime

class IngredientItem(BaseModel):
    name: str
    grams: float = Field(..., gt=0)

class IngredientNutrition(BaseModel):
    name: str
    source: str
    per_100g: Dict[str, float]
    grams: float
    total: Dict[str, float]
    ref_id: Optional[str] = None

class B1NutritionResponse(BaseModel):
    items: List[IngredientNutrition]
    totals: Dict[str, float]

class BarcodeResponse(BaseModel):
    code: str
    product_name: Optional[str] = None
    brand: Optional[str] = None
    nutriments_per_100g: Optional[Dict[str, float]] = None
    nutrition_per_package: Optional[Dict[str, float]] = None
    image_url: Optional[str] = None
    source: str = "openfoodfacts"


BACKEND_VERSION = "0.1.1"
FDC_API_KEY = os.getenv("FDC_API_KEY")


def kcal_bounds_check(kcal_per_100g: Optional[float]) -> Plausibility:
    if kcal_per_100g is None:
        return Plausibility(per_100g_kcal=None, within_bounds=False, message="No kcal/100g available")
    low, high = 10, 1000
    ok = low <= kcal_per_100g <= high
    msg = "OK" if ok else f"Out of bounds [{low},{high}]"
    return Plausibility(per_100g_kcal=kcal_per_100g, within_bounds=ok, message=msg)


def scale_totals(per_100g: Dict[str, float], grams: float) -> Dict[str, float]:
    factor = grams / 100.0
    return {k: round((per_100g.get(k, 0.0) * factor), 2) for k in ["kcal", "protein_g", "fat_g", "carbs_g"]}


# -----------------------------
# Root & health
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Food Vision API is running", "version": BACKEND_VERSION}


@app.get("/health")
def health():
    return {"ok": True, "version": BACKEND_VERSION}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            try:
                collections = db.list_collection_names()  # type: ignore[attr-defined]
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:100]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:100]}"

    return response


# -----------------------------
# Path A: Plate photo -> nutrition (mocked perception, real checks)
# -----------------------------
@app.post("/api/a1/estimate", response_model=A1Response)
async def estimate_meal(
    image: UploadFile = File(...),
    plate_diameter_cm: Optional[float] = Form(None),
):
    # Read bytes to mimic processing
    data = await image.read()

    # Simple size-derived portion heuristic (mock): bigger image bytes => slightly larger portion
    base_grams = 350.0
    size_factor = min(len(data) / (512 * 1024), 2.0)  # up to 2x for very large images
    portion_grams = round(base_grams * (0.8 + 0.2 * size_factor), 1)

    # Plate diameter hint refines portion +/- 20%
    if plate_diameter_cm and 20 <= plate_diameter_cm <= 34:
        scale = (plate_diameter_cm / 27.0)
        portion_grams = round(portion_grams * min(max(scale, 0.8), 1.2), 1)

    # Mock macro distribution: 40% carbs, 30% fat, 30% protein by kcal
    # 1g: carbs=4kcal, protein=4kcal, fat=9kcal
    kcal_per_100g_guess = 150.0  # plausible average cooked meal density
    total_kcal = round((portion_grams / 100.0) * kcal_per_100g_guess, 0)
    carbs_kcal = 0.4 * total_kcal
    fat_kcal = 0.3 * total_kcal
    protein_kcal = 0.3 * total_kcal

    carbs_g = round(carbs_kcal / 4.0, 1)
    fat_g = round(fat_kcal / 9.0, 1)
    protein_g = round(protein_kcal / 4.0, 1)

    plaus = kcal_bounds_check(kcal_per_100g_guess)

    estimation = MacroEstimation(
        kcal=total_kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        carbs_g=carbs_g,
        uncertainty=0.4,
        notes="Vision model not active in demo; values are heuristic."
    )

    payload = {
        "filename": image.filename,
        "size_bytes": len(data),
        "items": [],
        "portion_grams": portion_grams,
        "estimation": estimation.model_dump(),
        "plausibility": plaus.model_dump(),
        "created_at": datetime.now(timezone.utc)
    }

    try:
        create_document("estimationlog", payload)
    except Exception:
        pass

    return A1Response(**payload)


# -----------------------------
# Path B1: Ingredients -> nutrition totals (USDA/OFF mapping where possible)
# -----------------------------

def fetch_off_search(name: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"https://world.openfoodfacts.org/cgi/search.pl"
        params = {"search_terms": name, "search_simple": 1, "action": "process", "json": 1, "page_size": 1}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            js = r.json()
            if js.get("products"):
                return js["products"][0]
    except Exception:
        return None
    return None


def extract_nutrients_off(product: Dict[str, Any]) -> Dict[str, float]:
    nutr = product.get("nutriments", {})
    kcal = nutr.get("energy-kcal_100g") or (nutr.get("energy_100g") and nutr.get("energy_100g")/4.184) or 0
    return {
        "kcal": float(kcal or 0),
        "protein_g": float(nutr.get("proteins_100g") or 0),
        "fat_g": float(nutr.get("fat_100g") or 0),
        "carbs_g": float(nutr.get("carbohydrates_100g") or 0),
    }


def fetch_usda(name: str) -> Optional[Dict[str, Any]]:
    if not FDC_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"api_key": FDC_API_KEY, "query": name, "pageSize": 1},
            timeout=8,
        )
        if r.status_code == 200:
            js = r.json()
            foods = js.get("foods") or []
            if foods:
                food = foods[0]
                # Build per 100g nutrients from nutrients array if possible
                per100 = {"kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
                for n in food.get("foodNutrients", []):
                    name = (n.get("nutrientName") or "").lower()
                    unit = (n.get("unitName") or "").lower()
                    val = n.get("value")
                    if val is None:
                        continue
                    if "energy" in name and unit in ("kcal",):
                        per100["kcal"] = float(val)
                    elif "protein" in name:
                        per100["protein_g"] = float(val)
                    elif name in ("carbohydrate, by difference", "carbohydrate") or "carb" in name:
                        per100["carbs_g"] = float(val)
                    elif "total lipid" in name or "fat" in name:
                        per100["fat_g"] = float(val)
                food["per_100g"] = per100
                return food
    except Exception:
        return None
    return None


@app.post("/api/b1/nutrition", response_model=B1NutritionResponse)
def ingredients_nutrition(items: List[IngredientItem]):
    results: List[IngredientNutrition] = []

    for it in items:
        per100: Dict[str, float] = {"kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
        source = "heuristic"
        ref_id = None

        # Try USDA first (if key present)
        usda = fetch_usda(it.name)
        if usda and usda.get("per_100g"):
            per100 = usda["per_100g"]
            source = "usda"
            ref_id = str(usda.get("fdcId")) if usda.get("fdcId") else None
        else:
            # Fallback to OFF search
            off = fetch_off_search(it.name)
            if off:
                per100 = extract_nutrients_off(off)
                source = "openfoodfacts"
                ref_id = off.get("_id") or off.get("code")
            else:
                # Simple heuristic defaults if nothing found
                defaults = {
                    "chicken": {"kcal": 165, "protein_g": 31, "fat_g": 3.6, "carbs_g": 0},
                    "rice": {"kcal": 130, "protein_g": 2.7, "fat_g": 0.3, "carbs_g": 28},
                    "broccoli": {"kcal": 35, "protein_g": 2.6, "fat_g": 0.4, "carbs_g": 7},
                    "olive oil": {"kcal": 884, "protein_g": 0, "fat_g": 100, "carbs_g": 0},
                }
                for key, val in defaults.items():
                    if key in it.name.lower():
                        per100 = val
                        source = "default"
                        break

        totals = scale_totals(per100, it.grams)
        results.append(IngredientNutrition(name=it.name, source=source, per_100g=per100, grams=it.grams, total=totals, ref_id=ref_id))

    # Aggregate totals
    agg = {"kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
    for r in results:
        for k in agg.keys():
            agg[k] += r.total.get(k, 0.0)
    agg = {k: round(v, 2) for k, v in agg.items()}

    # Log to DB
    try:
        create_document("nutritioncalc", {"items": [i.model_dump() for i in results], "totals": agg, "created_at": datetime.now(timezone.utc)})
    except Exception:
        pass

    return B1NutritionResponse(items=results, totals=agg)


# -----------------------------
# Path B2: Barcode -> product from OFF, fallback USDA
# -----------------------------
@app.get("/api/b2/barcode/{code}", response_model=BarcodeResponse)
def barcode_lookup(code: str):
    # Open Food Facts lookup
    try:
        r = requests.get(f"https://world.openfoodfacts.org/api/v2/product/{code}.json", timeout=8)
        if r.status_code == 200:
            js = r.json()
            status = js.get("status")
            if status == 1:
                p = js.get("product", {})
                nutr = extract_nutrients_off(p)
                resp = BarcodeResponse(
                    code=code,
                    product_name=p.get("product_name"),
                    brand=(p.get("brands") or "").split(",")[0] if p.get("brands") else None,
                    nutriments_per_100g=nutr,
                    nutrition_per_package=None,
                    image_url=p.get("image_url"),
                )
                # Log
                try:
                    create_document("scanlog", {"type": "barcode", "code": code, "product": resp.model_dump(), "created_at": datetime.now(timezone.utc)})
                except Exception:
                    pass
                return resp
    except Exception:
        pass

    # Fallback USDA by code isn't practical; return not found minimal info
    return BarcodeResponse(code=code, product_name=None, brand=None, nutriments_per_100g=None, nutrition_per_package=None, image_url=None)


# -----------------------------
# Simple recipe suggestion from ingredients (rule-based JSON)
# -----------------------------
class RecipeRequest(BaseModel):
    title_hint: Optional[str] = None
    ingredients: List[IngredientItem]
    diet_tags: Optional[List[str]] = None

class RecipeResponse(BaseModel):
    title: str
    ingredients: List[Dict[str, Any]]
    steps: List[str]
    tags: List[str]

@app.post("/api/b1/recipe", response_model=RecipeResponse)
def recipe_from_ingredients(req: RecipeRequest):
    names = [i.name for i in req.ingredients]
    grams = [i.grams for i in req.ingredients]
    title = req.title_hint or "Schnelles Pfannengericht"

    if any("pasta" in n.lower() or "spaghetti" in n.lower() for n in names):
        title = "Einfaches Pasta-Gericht"
    if any("rice" in n.lower() for n in names):
        title = "Reis-Bowl"
    if any("egg" in n.lower() or "ei" in n.lower() for n in names):
        title = "Proteinreiches Rührei"

    steps = [
        "Zutaten vorbereiten: waschen, schneiden, abmessen.",
        "Pfanne/Topf erhitzen, ggf. Öl hinzufügen.",
        "Hauptzutat anbraten/kochen, dann restliche Zutaten hinzufügen.",
        "Mit Salz, Pfeffer und Gewürzen abschmecken.",
        "Servieren und genießen.",
    ]

    tags = list(set((req.diet_tags or []) + ["easy", "15min"]))

    return RecipeResponse(
        title=title,
        ingredients=[{"name": n, "grams": g} for n, g in zip(names, grams)],
        steps=steps,
        tags=tags,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
