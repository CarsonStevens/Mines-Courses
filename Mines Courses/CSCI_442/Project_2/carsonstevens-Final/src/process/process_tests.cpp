/**
 * This file contains tests for the Process class.
 *
 * You need to implement Process so that these tests pass.
 */

#include "process/process.h"
#include "page/page.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace std;


const string PROCESS_IMAGE =
    "AY3SmKknrmqdulbnXYZRtXnuQ571pA5HXAeB8qh0qR6n6yo303tdkAO9fZYWPLX3"
    "1L hUt1yI8nL4EkC0NMNm8pSKdVu5m7qDvfbXdHCG2ItC1BBMUG0i3IyfreBzxQG"
    "F7Lj9hyKVHtwlWg YZeXyaS04kzooqVjgqdH-3vK4osXWe912XGkIyeOaxgf0R7R"
    "PGR rmmlKuFNjamhMIGFl3hvECvH8E2LZC5a3TqTgtENEHSBHwbT8bxXwLU0eeJ4"
    "ixP1YqIOtbJiP-vipFGAxvSHMNFCZtpPfsbjG8iSjHs1Jg AUalfeVsguyfO5thx"
    "zq55qO8hEvBiZIy7ZLgXN7so2UdoJUYiIFApWNcq9gD4mjY1Nfc-l1jatMiQ7gqB"
    "acuYN4UgCLvA3WCfIjy1hqQvpaIaSo-JFPOT4FKM7Z22hTmPd31bBWbvuRX 52Ed"
    "hDEv4GHRCDJ8fJp4mAyliPHlJlIpzq OmtUT6Edbr-iphkUtr1DLbaHX8BZTlcud"
    "l4xbRwro 6AkKyvFkhv7Y4MxL0LimaID7OYb8cQ1eHS9 g pq3WfnnV8rpXvj23w"
    "GwSa0Sr-NB 8-GIVfPcn9WeFV36lH1YMJ3kRk6EwUlapkZo9BWCRH5dWM13E0-3o"
    "U6KtyOrLCqAVqbFx3o5xOTHJl91jiiJws3d Jom2UBZFNZV3kMTNjdHcno7LfrD3"
    "Bc7BSElu7D0SzQTvTu1cQvtFZU8dbgLMyl-AzxP86wqPmc2CMzx0gQ46CjxIhYPk"
    "dFTcj-QkTRlaEKgWAK9GYGbFgtJxQ2dDHFR35o6q-m0qesccJzBgg-0zIzltDdqa"
    "vgwb4J2mpVYj0 1uMnCTaDhRrlyIJQTRplqWknlls0G8dZ4KBtQLn-bKZdwzDgjg"
    "baSlKsgCadFff7Ae6Vo9lqxohpDiIImnCVjtVZHelavYOVp2lzxCRbAOc3gb9KRg"
    "1owqxRiFKy1zwdPRmxFlEpidBbppZpcOKuOZAlKBWMAJz8n8F1It6mqZZMoQd qi"
    "oWCC9Dq1ZrDgrSVwEaLHqfjvv5yCK4qui9D c5bGQTMLX6Zz0kt3Y-28qWl kU5X"
    "b9-aYV kenHP1OVYh CuNezqvJec32k34ZJcfst0YlIktWjmGeEBjmT5CybgQbD7"
    "sBbvohdU9FLbEPbH18BilcFDaNEB7Fg9y2ocTpNfsikp67Xbx8ijbUrkoWt2jsFC"
    "i7xbB8pazI4bnHxKjTiE3r-E4nY2bPvo  GV5cFOS5eblqyfCuVEMvXYXZ-S0ofm"
    "d2OJSwzgOuAd02m8V0VZLxzHe47I3SUHm7azLH9UVBqsuv0m0M8ClQNOU9TATlA "
    "-4uLkmo2KMWX2CPLy ajOOcCILJbWj3q4JtUE9hcRfF7NUrZfcRaoDdZF icJpxG"
    "lFnz cxH2IpRWnYWKDjeqWNGi U-Nbfjs2vGNWxZL-kd5Wawk2diuqgdve0MfcrD"
    "P737ywmI0-O4TnCMM96fl-aGSst00AA6HRJ8ByN5hawRh0lFeFfFu7JxK-ElgnqX"
    "1h5VEbJXiEuQab2NxwUvehLy rTWLB9iAfHWb0iVYOPoYnHeeEISF-4ux3Kp SP1"
    "UnrbmUWV8a6C4AZrZLVnuoWCtTCqvzAsaXQ5mWujYt24ecTG1Los1RybVhF 38tt"
    "wUolQbkLZSkP3VUK9q EUYkFlWO fmb-bnRuz-SrZ5f0zd nEr07LFrQwfVveDwQ"
    "gBelo-hJnYT1Ub0xmGQVs7N N3f DuhoBhcJ-McXmJAS 6h bJTpmXMiBmYnDlXj"
    "0wuqzr90ZTo69s5ypwfFAr74cRwxURYtqgfUlJoZsHypQFlEhgWQKL6GRurmfRL6"
    "JykQn5Vi16jCGf8LtcvDzjg Mg2n SUYRelPKsdR5PVdarmgx3wLq0c5dmLqjBzd"
    "31nj3FHF1ROp4umCXLruU1JR YBwTP!Pw2p4i-oTnUWL3BXVTJJVeoc0UHMB CsO"
    "FOR GREAT JUSTICE!";


Process* read_process() {
  istringstream input_stream(PROCESS_IMAGE);

  return Process::read_from_input(input_stream);
}


TEST(Process, TotalSize) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  ASSERT_EQ(2002, process->size());
}


TEST(Process, NumPages) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  ASSERT_EQ(32, process->pages.size());
}


TEST(Process, IsValidPage_ValidIndex) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  ASSERT_TRUE(process->is_valid_page(31));
}


TEST(Process, IsValidPage_InvalidIndex) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  ASSERT_FALSE(process->is_valid_page(32));
}

//FIX
TEST(Process, NonLastPageSizes) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  ASSERT_FALSE(process->pages.empty()) << "Process has no pages";

  for (size_t i = 0; i < process->pages.size() - 1; i++) {
    ASSERT_NE(nullptr, process->pages[i]);
    EXPECT_EQ(Page::PAGE_SIZE, process->pages[i]->size());
  }
}

//FIX
TEST(Process, LastPageSize) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);
  ASSERT_FALSE(process->pages.empty()) << "Process has no pages";
  ASSERT_NE(nullptr, process->pages.back());
  EXPECT_EQ(18, process->pages.back()->size());
}

//FIX
TEST(Process, PageContent) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  stringstream expected_bytes;
  
  for (size_t i = 0; i < 31; i++) {
    ASSERT_TRUE(process->is_valid_page(i));
    ASSERT_NE(nullptr, process->pages[i]);
    ASSERT_TRUE(process->pages[i]->is_valid_offset(i));

    expected_bytes << process->pages[i]->get_byte_at_offset(i);
  }

  ASSERT_EQ(expected_bytes.str(), "ALL YOUR BASE ARE BELONG TO US!");
}


TEST(Process, GetRss) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  for (size_t i = 3; i < 7; i++) {
    process->page_table.rows[i].present = true;
  }

  ASSERT_EQ(4, process->get_rss());
}


TEST(Process, GetFaultPercent) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  process->memory_accesses = 50;
  process->page_faults = 32;

  ASSERT_DOUBLE_EQ(64.0, process->get_fault_percent());
}


TEST(Process, GetFaultPercent_ZeroDenominator) {
  Process* process = read_process();
  ASSERT_NE(nullptr, process);

  process->memory_accesses = 0;
  process->page_faults = 0;

  ASSERT_DOUBLE_EQ(0.0, process->get_fault_percent());
}
