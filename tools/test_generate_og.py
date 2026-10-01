"""Regression checks for social-card source metadata and rendering inputs."""

import pathlib
import unittest

import generate_og


ROOT = pathlib.Path(__file__).resolve().parent.parent


class CardSourceTests(unittest.TestCase):
    def test_every_shared_page_has_its_own_card(self):
        for section in ("_posts", "_pages", "_projects"):
            for path in sorted((ROOT / section).glob("*.md")):
                with self.subTest(path=path.name):
                    slug = generate_og.slugify(path)
                    if slug == "mcp-grew-up-fast":
                        slug += "-v2"  # new URL breaks social-platform image caches
                    card = f"/assets/images/og/{slug}.jpg"
                    source = path.read_text(encoding="utf-8")
                    self.assertTrue(f"og_image: {card}" in source, f"{path}: missing {card}")
                    self.assertTrue((ROOT / card.lstrip("/")).is_file())

    def test_homepage_card_has_new_asset_url(self):
        config = (ROOT / "_config.yml").read_text(encoding="utf-8")
        self.assertTrue('og_image                 : "/assets/images/og/site-card-v2.jpg"' in config)
        self.assertTrue((ROOT / "assets/images/og/site-card-v2.jpg").is_file())

    def test_card_title_matches_rendered_smart_apostrophe(self):
        title, _ = generate_og.parse_front_matter(
            ROOT / "_posts" / "2025-10-09-architecs-anthill-ai.md"
        )
        self.assertIn("Prize’s Lesson", title)

    def test_project_title_without_trailing_newline(self):
        title, date = generate_og.parse_front_matter(
            ROOT / "_projects" / "volatility-smile.md"
        )
        self.assertEqual(title, "Volatility Smile Explorer")
        self.assertEqual(date, "July 25, 2025")


if __name__ == "__main__":
    unittest.main()
