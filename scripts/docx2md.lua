-- Pandoc Lua filter used by docx2md.py to turn the book .docx into clean GFM.
-- Run with: pandoc -f docx+styles --lua-filter docx2md.lua

local DROP_STYLES = { ["toc 1"] = true, ["toc 2"] = true, ["toc 3"] = true }
local BLOCK_STYLES = { Code = "code", ["HTML Code"] = "code", Output = "output" }

-- Flatten inlines to plain text, keeping explicit line breaks.
local function to_text(inlines)
  local out = {}
  for _, el in ipairs(inlines) do
    if el.t == "Str" then
      table.insert(out, el.text)
    elseif el.t == "Space" or el.t == "SoftBreak" then
      table.insert(out, " ")
    elseif el.t == "LineBreak" then
      table.insert(out, "\n")
    elseif el.t == "Code" or el.t == "RawInline" then
      table.insert(out, el.text)
    elseif el.t == "Quoted" then
      local q = el.quotetype == "SingleQuote" and "'" or '"'
      table.insert(out, q .. to_text(el.content) .. q)
    elseif el.content then
      table.insert(out, to_text(el.content))
    end
  end
  return (table.concat(out):gsub("\u{a0}", " "))
end

local function block_text(blk)
  if blk.t == "Para" or blk.t == "Plain" then
    return to_text(blk.content)
  elseif blk.t == "OrderedList" then
    local lines = {}
    for i, item in ipairs(blk.content) do
      local parts = {}
      for _, b in ipairs(item) do table.insert(parts, block_text(b)) end
      table.insert(lines, (blk.start + i - 1) .. ". " .. table.concat(parts, "\n"))
    end
    return table.concat(lines, "\n")
  end
  return pandoc.utils.stringify(blk)
end

local function guess_lang(code)
  local first = code:match("^%s*(%S+)") or ""
  if first:match("^%$") or first == "pip" or first == "streamlit" or first == "curl"
      or first == "git" or first == "ollama" or first == "python" then
    return "bash"
  elseif first:match("^[{%[]") then
    return "json"
  elseif first:match("^<") then
    return "html"
  elseif code:match("^%s*name:") or code:match("^%s*on:") then
    return "yaml"
  end
  return "python"
end

-- A few snippets in the docx were left in the Normal style.
local CODE_PARA = { "^python %-m ", "^pip install ", "^\\?# ", "^env\\Scripts", "^prompt = ", "^img = " }

local function looks_like_code(blk)
  if blk.t ~= "Para" then return false end
  local text = to_text(blk.content)
  for _, pat in ipairs(CODE_PARA) do
    if text:match(pat) then return true end
  end
  return false
end

local function style_of(blk)
  if blk.t == "Div" and blk.attributes["custom-style"] then
    return blk.attributes["custom-style"]
  end
end

-- Merge runs of consecutive Code/Output paragraphs into single code blocks.
function Blocks(blocks)
  local out, buf, kind = {}, {}, nil
  local function flush()
    if kind then
      local code = table.concat(buf, "\n"):gsub("%s+$", "")
      local lang = guess_lang(code)
      if kind == "output" and lang ~= "bash" then lang = "text" end
      table.insert(out, pandoc.CodeBlock(code, { class = lang }))
    end
    buf, kind = {}, nil
  end
  for _, blk in ipairs(blocks) do
    local style = style_of(blk)
    local k = style and BLOCK_STYLES[style] or (looks_like_code(blk) and "code")
    local _, quotes = table.concat(buf, "\n"):gsub('"""', "")
    if kind == "code" and not style and quotes % 2 == 1 then
      -- inside an unterminated triple-quoted string
      table.insert(buf, block_text(blk))
    elseif k == "code" and not style then
      if kind and kind ~= k then flush() end
      kind = k
      table.insert(buf, to_text(blk.content))
    elseif k then
      if kind and kind ~= k then flush() end
      kind = k
      for _, inner in ipairs(blk.content) do
        table.insert(buf, block_text(inner))
      end
    elseif style and DROP_STYLES[style] then
      -- skip table of contents entries
    else
      flush()
      if style then
        for _, inner in ipairs(blk.content) do table.insert(out, inner) end
      else
        table.insert(out, blk)
      end
    end
  end
  flush()
  return out
end

-- Unwrap character styles (Hyperlink, footnote reference, ...).
function Span(el)
  if el.attributes["custom-style"] then
    return el.content
  end
end

local function clean_image(img, alt)
  img.attributes = {}
  img.title = ""
  img.src = img.src:gsub("^%./", "")
  if alt then
    img.caption = alt
  elseif pandoc.utils.stringify(img.caption):match("automatically generated") then
    img.caption = {}
  end
  return img
end

function Image(img)
  return clean_image(img)
end

-- Figures become an image followed by an italic caption line.
function Figure(fig)
  local caption = pandoc.utils.blocks_to_inlines(fig.caption.long)
  local imgs = {}
  fig.content:walk({ Image = function(img) table.insert(imgs, clean_image(img, caption)) end })
  local blocks = {}
  for _, img in ipairs(imgs) do
    table.insert(blocks, pandoc.Para({ img }))
  end
  if #caption > 0 then
    table.insert(blocks, pandoc.Para({ pandoc.Emph(caption) }))
  end
  return blocks
end

-- Drop empty headings left over from page-break tricks.
function Header(h)
  if pandoc.utils.stringify(h.content):match("^%s*$") then
    return {}
  end
  h.attr = pandoc.Attr()
  h.content = h.content:walk({ LineBreak = function() return pandoc.Space() end })
  return h
end

-- Drop table styles (GFM cannot render them) and show captions in italics.
function Table(tbl)
  tbl.attr = pandoc.Attr()
  -- Promote the first row to a header when the docx table has none
  local body = tbl.bodies[1]
  if #tbl.head.rows == 0 and body and #body.body > 1 then
    tbl.head.rows = { table.remove(body.body, 1) }
  end
  local caption = pandoc.utils.blocks_to_inlines(tbl.caption.long)
  if #caption == 0 then return tbl end
  tbl.caption = { long = {} }
  return { tbl, pandoc.Para({ pandoc.Emph(caption) }) }
end
