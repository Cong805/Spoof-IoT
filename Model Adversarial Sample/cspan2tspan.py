#!/usr/bin/env python3
# -*- coding:utf-8 -*-


def get_tok2char_span_map(text):
    '''
    get character level span for each token
    '''
    tokenize = lambda text: text.split(" ")
    tokens = tokenize(text)
    # print(tokens)
    tok2char_span = []
    char_num = 0
    for tok in tokens:
        tok2char_span.append((char_num, char_num + len(tok)))
        char_num += len(tok) + 1  # +1: whitespace
    return tok2char_span


def get_char2tok_span(text):
    '''
    map character index to token level span
    '''
    tok2char_span = get_tok2char_span_map(text)
    # print(tok2char_span)
    char_num = None
    for tok_ind in range(len(tok2char_span) - 1, -1, -1):
        # print(tok_ind)
        if tok2char_span[tok_ind][1] != 0:
            char_num = tok2char_span[tok_ind][1]
            break
    char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
    for tok_ind, char_sp in enumerate(tok2char_span):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
            if tok_sp[0] == -1:
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1
    return char2tok_span


def char_span2tok_span(char_span, char2tok_span):
    '''
    map character level span to token level span
    '''
    tok_span_list = char2tok_span[char_span[0]:char_span[1]]
    tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
    return tok_span


# def preprocess(input):
#     # input -> token list
#     text = utils.extracTextFromHtml(input)
#     text = text.replace("/", " ").replace("=", " ").replace(":", "").replace("\"", "").replace("'", " ")
#     token_list = text.split()
#     return token_list
#
# def htmlTag_find_text_index(soup, html_tag, html_doc):
#     region_index = []
#     for head in soup.find_all(html_tag):
#         text = head.get_text()
#         # region_index.extend(text.split())
#         text = " ".join(preprocess(text))
#         if text:
#             text = text.replace("(", r"\(").replace(")", r"\)").replace("*", "\*").replace(r".", "\.").replace(r"+",
#                                                                                                                "\+").replace(
#                 r"?", "\?").replace(r"[", "\[").replace(r"]", "\]")
#
#             search_res = re.search(r'{}'.format(text), r'{}'.format(html_doc), re.IGNORECASE)
#             # print(search_res)
#             if search_res:
#                 char2tok_span = get_char2tok_span(html_doc)
#                 # print(char2tok_span)
#                 tok_span = char_span2tok_span(search_res.span(), char2tok_span)
#                 # print(tok_span,html_doc.split()[62])
#                 # exit()
#                 # start_index = search_res.span()[0]
#                 # end_index = search_res.span()[1]
#                 region_index.extend(list(range(tok_span[0], tok_span[1])))
#     return region_index
#
# def region_split_VFA(soup, banner_rm_html):
#     region_VFA = []
#     html_tag_list = ["h1", "h2", "h3", "h4", "h5", "h6", "b", "big", "em", "strong", "i", "u", "title"]
#     for html_tag in html_tag_list:
#         region_VFA.extend(htmlTag_find_text_index(soup, html_tag, banner_rm_html))
#         # content = similar_replace(soup, title_items, title_text)
#     return region_VFA

if __name__ == '__main__':
    # import re
    # import json
    # import random
    # import utils
    # from bs4 import BeautifulSoup, NavigableString

    demo = "HTTP/1.1 200 OK\r\nServer: EWS-NIC5/19.16\r\nConnection: Keep-Alive\r\nContent-Type: text/html\r\nContent-Length: 540\r\n\r\n<html>\r\n<head>\r\n<meta http-equiv=\"Content-Type\" content=\"text/html;charset=UTF-8\">\r\n<meta http-equiv=\"Content-Style-Type\" content=\"text/css\">\r\n<STYLE TYPE=\"text/css\"><!--\r\nbody { font-family: Arial, Helvetica, sans-serif, Verdana; margin-left:0; margin-right:0; margin-top:0; margin-bottom:0; }\r\n--></STYLE>\r\n<link rel=\"stylesheet\" href=\"common_dell.css\" type=\"text/css\">\r\n<title>Dell C3760dn Color Laser</title>\r\n</head>\r\n<a href=\"http://www.dell.com \" target=_blank><img src=\"images/delllogo2.png\" alt=\"\" border=\"0\"></a></body>\r\n</html>\r\n"
    # demo = "HTTP/1.1 200 Ok\r\nServer: micro_httpd\r\nCache-Control: no-cache\r\nDate: Wed, 04 Dec 2019 10:18:57 GMT\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n<html>\n<head>\n<meta http-equiv=\"Pragma\" content=\"no-cache\" charset='utf-8'/>\n<title>300Mbps Wireless N ADSL2+ Modem Router TD-W8960N</title>\n<style type=\"text/css\">\nbody {\n\tfont-family: Arial,sans-serief;\n\tbackground-color: #E5E5E5;\n\tmargin: 0px;\n\tpadding: 0px;\n}\ndiv.loginBox {\n\tdisplay: block;\n\tposition: relative;\n\tmargin-top: 10%;\n\ttext-align: center;\n}\n.noteDiv {\n\tcolor: gray;\n\tfont-family: Arial;\n\twidth: 395px;\n\ttext-align: left;\n\tmargin: 0px auto;\n\tfont-size: 14px;\n}\ndiv.panelThre\n{\n\tmargin-top:10px;\n}\ndiv.picDiv {\n\twidth: 395px;\n\theight: 276px;\n\tbackground: url(login_bg2.png);\n\tposition: relative;\n}\ninput.pcPassword {\n\twidth: 300px;\n\theight: 50px;\n\tline-height: 50px;\n\tpadding-left: 20px;\n}\nul {\n\tpadding: 60px 0px 0px 0px;\n\tmargin: 0px;\n\tlist-style: none;\n}\nul li {\n\theight: 34px;\n\twidth: 222px;\n}\nli.unLi {\n\tbackground: url(login_idunselect.png);\n}\nli.pwLi {\n\tbackground: url(login_pwunselect.png);\n}\nli.blank {\n\theight: 15px;\n}\nbutton.loginBtn {\n\theight: 34px;\n\tdisplay: inline-block;\n\twidth: 113px;\n\tborder: 0px;\n\tmargin-top: 30px;\n\tbackground: url(login.png);\n\tcursor: pointer;\n}\ninput.text {\n\tborder: 0px;\n\theight: 26px;\n\tline-height: 26px;\n\twidth: 175px;\n\tpadding: 0px;\n\tmargin: 4px 0px 0px 30px;\n\tfont-size: 14px;\n\tcolor: #6a6969;\n\tfont-family: Verdana,Arial;\n\tfont-weight: regular;\n\toutline: none; /* For chrome textbox border */\n}\n#ban {\n\theight: 90px;\n\tbackground: #09C url(2x87-lan.jpg) repeat-x;\n}\n</style>\n<script type=\"text/javascript\">\n/* BY yl */\nvar noPwd = false;\nvar authTimes = 0;\nvar displayBan = 1;\nauthTimes = 0;\ndisplayBan = 1;\nvar isLocked = false;\nfunction Base64Encoding(input) {\n\tvar keyStr = \"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\";\n\tvar output = \"\";\n\tvar chr1, chr2, chr3, enc1, enc2, enc3, enc4;\n\tvar i = 0;\n\tinput = utf8_encode(input);\n\twhile (i < input.length) {\n\t\tchr1 = input.charCodeAt(i++);\n\t\tchr2 = input.charCodeAt(i++);\n\t\tchr3 = input.charCodeAt(i++);\n\t\tenc1 = chr1 >> 2;\n\t\tenc2 = ((chr1 & 3) << 4) | (chr2 >> 4);\n\t\tenc3 = ((chr2 & 15) << 2) | (chr3 >> 6);\n\t\tenc4 = chr3 & 63;\n\t\t\n\t\tif (isNaN(chr2)) {\n\t\t\tenc3 = enc4 = 64;\n\t\t} else if (isNaN(chr3)) {\n\t\t\tenc4 = 64;\n\t\t}\n\t\toutput = output + keyStr.charAt(enc1) + keyStr.charAt(enc2) + keyStr.charAt(enc3) + keyStr.charAt(enc4);\n\t}\n\treturn output;\n}\nfunction utf8_encode(string) {\n\tstring = string.replace(/\\r\\n/g,\"\\n\");\n\tvar utftext = \"\";\n\tfor (var n = 0; n < string.length; n++) {\n\t\tvar c = string.charCodeAt(n);\n\t\tif (c < 128) {\n\t\t\tutftext += String.fromCharCode(c);\n\t\t} else if ((c > 127) && (c < 2048)) {\n\t\t\tutftext += String.fromCharCode((c >> 6) | 192);\n\t\t\tutftext += String.fromCharCode((c & 63) | 128);\n\t\t} else {\n\t\t\tutftext += String.fromCharCode((c >> 12) | 224);\n\t\t\tutftext += String.fromCharCode(((c >> 6) & 63) | 128);\n\t\t\tutftext += String.fromCharCode((c & 63) | 128);\n\t\t}\n\t}\n\treturn utftext;\n}\nfunction PCWin(event) {\n\tif (event.keyCode == 13) {\n\t\tPCSubWin();\n\t}\n}\nfunction PCSubWin0() {\n\tvar pw = $('pcPassword'), pw2 = $('pcPassword2');\n\tvar pwd = pw.value;\n\tif (pwd == '' || pwd == 'admin' || pwd.indexOf(' ') != -1) {\n\t\talert('Nale\u017cy wprowadzi\u0107 has\u0142o. Has\u0142o nie mo\u017ce by\u0107 s\u0142owem admin ani zawiera\u0107 spacji!');\n\t\tpw.focus();\n\t\treturn false;\n\t}\n\tif (pwd != pw2.value) {\n\t\talert('Potwierdzenie has\u0142a musi by\u0107 takie samo jak nowe has\u0142o!');\n\t\tpw2.focus();\n\t\treturn false;\n\t}\n\treturn true;\n}\nfunction PCSubWin() {\n\tif ((noPwd && !PCSubWin0()) ||\n\t\t(!noPwd && isLocked)) {\n\t\treturn;\n\t}\n\tvar auth;\n\tvar password = $(\"pcPassword\").value;\n\tvar userName = $(\"userName\").value;\n\tauth = \"Basic \"+Base64Encoding(userName+\":\"+password);\n\tdocument.cookie = \"Authorization=\" + auth;\n\tlocation.reload();\n}\nfunction deleteCookie(name) {\n\tvar LargeExpDate = new Date();\n\tdocument.cookie = name + \"=; expires=\" +LargeExpDate.toGMTString();\n}\nfunction w(str) {\n\tdocument.write(str);\n}\nfunction $(id) {\n\treturn document.getElementById(id);\n}\nfunction lockWeb(bLock) {\n\t$(\"userName\").disabled = bLock;\n\t$(\"pcPassword\").disabled = bLock;\n}\nfunction pageLoad() {\n\tdeleteCookie(\"Authorization\");\n\tnoPwd ? pageLoad0() : pageLoad1();\n}\nfunction pageLoad0() {\n\tvar node = $('ulNode'), node2;\n\tnode.innerHTML += '<li class=\"blank\"></li><li id=\"pwLi2\" class=\"pwLi\"><input class=\"text\" id=\"pcPassword2\" type=\"password\" maxlength=\"15\"/></li>';\n\tnode.style.paddingTop = '30px';\n\tnode = $('userName'); node.value = 'admin'; node.disabled = true;\n\t$('tip').innerHTML = 'Wprowad\u017a has\u0142o (musi by\u0107 inne ni\u017c \"admin\").';\n\tnode = $('pcPassword'); node2 = $('pcPassword2');\n\tnode.onfocus = node2.onfocus = function() {\n\t\tthis.parentNode.style.background = 'url(login_pwselect.png)';\n\t};\n\tnode.onblur = node2.onblur = function() {\n\t\tthis.parentNode.style.background = 'url(login_pwunselect.png)';\n\t};\n\tnode.focus();\n}\nfunction pageLoad1() {\n\tvar tip = $(\"tip\"), tipStr=\"\", note = $(\"note\");\n\tvar unLi = $(\"unLi\");\n\tvar pwLi = $(\"pwLi\");\n\tvar userName = $(\"userName\");\n\tvar pcPassword = $(\"pcPassword\");\n\tvar banner = $(\"ban\");\n\tif (displayBan == 0)\n\t\tbanner.style.display = \"none\";\n\tpcPassword.onfocus = function() {\n\t\tpwLi.style.background = \"url(login_pwselect.png)\";\n\t};\n\tpcPassword.onblur = function() {\n\t\tpwLi.style.background = \"url(login_pwunselect.png)\";\n\t};\n\tuserName.onfocus = function() {\n\t\tunLi.style.background = \"url(login_idselect.png)\";\n\t};\n\tuserName.onblur = function() {\n\t\tunLi.style.background = \"url(login_idunselect.png)\";\n\t};\n\tif (authTimes > 0) {\n\t\tnote.innerHTML = \"Uwaga:\";\n\t\ttip.innerHTML = \"Nieprawid\u0142owa nazwa u\u017cytkownika lub has\u0142o, wprowad\u017a ponownie.\";\n\t}\n\tif (authTimes < 5) {\n\t\tuserName.focus();\n\t}\n\tif (authTimes >= 5) {\n\t\tisLocked = true;\n\t\tlockWeb(true);\n\t\tnote.innerHTML = \"Uwaga:\";\n\t\ttipStr = \"Przekroczono limit 5 nieudanych pr\u00f3b, spr\u00f3buj ponownie za \";\n\t\ttip.innerHTML = tipStr + expiredTime + \"s\";\n\t\twindow.setTimeout(function() {\n\t\t\tif (expiredTime <= 1) {\n\t\t\t\tnote.innerHTML = \"\";\n\t\t\t\ttip.innerHTML = \"\";\n\t\t\t\tisLocked = false;\n\t\t\t\tlockWeb(false);\n\t\t\t\tuserName.focus();\n\t\t\t\treturn ;\n\t\t\t}\n\t\t\texpiredTime--;\n\t\t\ttip.innerHTML = tipStr + expiredTime + \"s\";\n\t\t\twindow.setTimeout(arguments.callee, 1000);\n\t\t}, 1000);\n\t}\n}\n</script>\n</head>\n<body onload=\"pageLoad()\" onkeypress=\"PCWin(event)\">\n<div id=\"ban\">\n\t<a href=\"http://www.tp-link.com\"><img id=\"logo\" src=\"280x87-lan.jpg\" style=\"float:left;\" border=\"0\"/></a>\n\t<div id=\"mod\" style=\"height:87px; margin:0; text-align:right; overflow:hidden; background:url(490x87-lan.jpg) no-repeat right;\">\n\t\n\t<div style=\"font-weight:bold; font-size:14pt; color:#ffffff; margin-top:27px; margin-right:30px\"  align='right'>Bezprzewodowy router/modem ADSL2+, standard N, 300Mb/s</div>\n\t<div style=\"font-weight:bold; font-size:11pt; color:#ffffff; margin-top:5px; margin-right:30px\" align='right'>Model TD-W8960N</div>\n\t</div>\n\t<img id=\"bhr\" src=\"770x3-lan.jpg\" style=\"height:3px; width:100%; display:block;\"/>\n</div>\n<div class=\"loginBox\">\n\t<div class=\"noteDiv\">\n\t\t<span id=\"note\"></span>\n\t\t<span id=\"tip\"></span>\n\t</div>\n\t<div class=\"panelThre\" align=\"center\">\n\t\t<div class=\"picDiv\">\n\t\t\t<ul id=\"ulNode\">\n\t\t\t\t<li id=\"unLi\" class=\"unLi\"><input class=\"text\" id=\"userName\" type=\"input\" maxlength=\"16\"/></li>\n\t\t\t\t<li class=\"blank\"></li>\n\t\t\t\t<li id=\"pwLi\" class=\"pwLi\"><input class=\"text\" id=\"pcPassword\" type=\"password\" maxlength=\"16\"/></li>\n\t\t\t</ul>\n\t\t\t<button type=\"button\" class=\"loginBtn\" onclick=\"PCSubWin();return false\"></button>\n\t\t\t<div style=\"font-size: 10px; margin-top: 20px;\">\n\t\t\t\t<label id=\"copyright\" >Copyright &copy;2016 TP-LINK Technologies Co., Ltd. Wszelkie prawa zastrze\u017cone.</label>\n\t\t\t</div>\n\t\t</div>\n\t</div>\n</div>\n</body>\n</html>\n"
    char2tok_span = get_char2tok_span(demo)
    print(char2tok_span)
    tok_span = char_span2tok_span([0, 7], char2tok_span)
    print(tok_span)

    # soup = BeautifulSoup(demo, 'html.parser')
    # # print(soup)
    # token_list = preprocess(demo)
    # banner_rm_html = ' '.join(token_list)
    # # print(banner_rm_html)
    # VFA_index = region_split_VFA(soup, banner_rm_html)
    # print("VFA_index", VFA_index)
    # region_other = []
    # if soup.find_all("body"):
    #     a = soup.body.descendants
    #     for i in a:
    #         # print(i)
    #         if not isinstance(i, NavigableString):
    #             i_text = i.get_text()
    #             # region_other.extend(i_text.split())
    #             i_text = " ".join(preprocess(i_text))
    #             # print(i_text)
    #             if i_text:
    #                 i_text = i_text.replace("(", r"\(").replace(")", r"\)").replace("*", "\*").replace(r".",
    #                                                                                                    "\.").replace(
    #                     r"+", "\+").replace(r"?", "\?").replace(r"[", "\[").replace(r"]", "\]")
    #                 search_res = re.search(r'{}'.format(i_text), r'{}'.format(banner_rm_html), re.IGNORECASE)
    #                 print(search_res)
    #                 if search_res:
    #                     char2tok_span = get_char2tok_span(banner_rm_html)
    #                     # print(char2tok_span)
    #                     tok_span = char_span2tok_span(search_res.span(), char2tok_span)
    #                     # print(tok_span)
    #                     region_other.extend(list(range(tok_span[0], tok_span[1])))
    #                     # start_index = search_res.span()[0]
    #                     # end_index = search_res.span()[1]
    #                     # region_other.extend(list(range(start_index, end_index + 1)))
    #                 # content = level4_replace(soup, i, i_text)
    # print(region_other)
