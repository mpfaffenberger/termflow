class TestFromPalette:
    def test_derives_accents_from_ansi_slots(self):
        from termflow.render.style import RenderStyle
        from termflow.themes import get_palette

        pal = get_palette("purple_puppy")
        style = RenderStyle.from_palette(pal)
        assert style.bright == pal.ansi[12]
        assert style.symbol == pal.ansi[5]
        assert style.error == pal.ansi[9]
        assert style.dark == pal.bg

    def test_accepts_plain_dict(self):
        from termflow.render.style import RenderStyle

        ansi = [f"#0000{i:02x}" for i in range(16)]
        style = RenderStyle.from_palette({"ansi": ansi, "bg": "#101010"})
        assert style.bright == ansi[12]
        assert style.grey == ansi[8]
        assert style.dark == "#101010"

    def test_missing_slots_fall_back_to_defaults(self):
        from termflow.render.style import RenderStyle

        base = RenderStyle.default()
        style = RenderStyle.from_palette({"ansi": []})
        assert style.bright == base.bright
        assert style.error == base.error
