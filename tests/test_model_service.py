"""API parity against the locally trained v001 bundle."""

import unittest
from unittest.mock import patch

import httpx
import numpy as np
import torch

from recommender_app.ml.features import reader_features
from recommender_app.model_service.api import DEFAULT_BUNDLE, create_app


class ModelServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_bundle_api_parity(self):
        app = create_app(DEFAULT_BUNDLE)
        with patch.dict("os.environ", {"MODEL_API_KEY": "parity-key"}):
            async with (app.router.lifespan_context(app),
                        httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                          base_url="http://test") as client):
                offline = app.state.predictor
                ids = list(offline.books)
                rated = dict(zip(ids[:4], [1, 3, 4, 5]))
                rated["outside-catalog"] = 4
                selected = ids[4:7]
                headers = {"X-API-Key": "parity-key"}
                self.assertEqual((await client.get("/health")).json()["model_version"], offline.version)
                self.assertEqual((await client.post("/v1/predict", json={"book_ids": selected})).status_code, 401)

                known = next(iter(offline.users))
                with np.load(DEFAULT_BUNDLE / "catalog.npz", allow_pickle=False) as catalog:
                    reserved = next(str(i) for i in catalog["user_ids"] if str(i) not in offline.users)
                for reader, expected_index in ((known, offline.users[known]),
                                               (None, offline.model.unknown_user),
                                               (reserved, offline.model.unknown_user)):
                    with self.subTest(reader=reader):
                        response = await client.post("/v1/predict", headers=headers, json={
                            "reader_id": reader, "ratings": rated, "book_ids": selected,
                        })
                        self.assertEqual(response.status_code, 200, response.text)
                        predictions = response.json()["predictions"]
                        self.assertEqual([p["book_id"] for p in predictions], selected)
                        manifest = offline.manifest
                        profile = reader_features(list(rated.values()), mean=manifest["mean_rating"],
                                                  default_std=manifest["default_reader_std"],
                                                  scaler=manifest["scaler"])
                        indices = np.array([offline.books[book] for book in selected])
                        features = offline.book_features[indices]
                        features[:, manifest["user_columns"]] = profile
                        with torch.inference_mode():
                            direct = offline.model(torch.from_numpy(features),
                                                   torch.full((len(indices),), expected_index),
                                                   torch.from_numpy(indices)).clamp(1, 5).numpy()
                        np.testing.assert_allclose([p["predicted_rating"] for p in predictions],
                                                   direct, atol=1e-5, rtol=0)

                response = await client.post("/v1/recommend", headers=headers, json={
                    "reader_id": known, "ratings": rated, "exclude_book_ids": selected[:1], "top_n": 10,
                })
                self.assertEqual(response.status_code, 200, response.text)
                results = response.json()["predictions"]
                candidates = [book for book in ids if book not in rated and book != selected[0]]
                scores = offline.score(candidates, rated, known)
                expected = sorted(zip(candidates, scores), key=lambda row: -row[1])[:10]
                self.assertEqual([p["book_id"] for p in results], [book for book, _ in expected])
                np.testing.assert_allclose([p["predicted_rating"] for p in results],
                                           [score for _, score in expected], atol=1e-5, rtol=0)
                self.assertEqual((await client.post("/v1/predict", headers=headers,
                                             json={"book_ids": ["outside-catalog"]})).status_code, 404)
                self.assertEqual((await client.post("/v1/recommend", headers=headers,
                                             json={"ratings": {"outside-catalog": 0}})).status_code, 422)
